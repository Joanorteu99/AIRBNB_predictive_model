import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import re 
import seaborn as sns
from scipy import stats

# Eliminate warnings for presentation purpuses
import warnings
warnings.filterwarnings('ignore') #ignore all warnings, very messy
#warnings.filterwarnings('default')
pd.options.mode.chained_assignment = None
from rich import print

#preatty printing
from rich import print

from IPython.display import display, HTML
pd.set_option('display.max_colwidth', 100)
from prettytable import PrettyTable
import icecream as ic
import os

import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import ast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-------------------------------------------- CUSTOM TRANSFORMER AMENITIES ---------------------------------------
#-----------------------------------------------------------------------------------------------------------------

class AmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encode_dict = {}             # This contains the dictionary of transformations
        self.filtered_amenities_list = [] # This contains the list of unique features ONEHOTENCODED

    def fit(self, X, y=None):
        all_amenities = []

        # Extract and preprocess amenities
        for sublist in X['amenities']:
            real_sublist = ast.literal_eval(sublist)
            for word in real_sublist:
                all_amenities.append(word)

        unique_amenities_list = sorted(set(all_amenities))

        # Compress specific amenities
        patterns_to_compress = [
            ('TV', 'TV'), ('shampoo', 'shampoo'), ('soap', 'soap'), ('parking', 'parking', 'garage'),
            ('exercise', 'gym'), ('stove', 'stove'), ('oven', 'oven'), ('coffee', 'coffee'), ('refrigerator', 'refrigerator'),
            ('books', 'books'), ('pool', 'pool'), ('clothing', 'clothing'), ('Washer', 'washer'), ('conditioner', 'conditioner'),
            ('dryer', 'dryer'), ('backyard', 'backyard'), ('Baby', 'Baby'), ('console', 'console'), ('sound', 'sound'),  
            ('grill', 'grill'), ('fireplace', 'fireplace'), ('Shared hot tub', 'Shared hot tub'), ('hot tub', 'hot tub'), 
            ('beach', 'beach'), ('chair', 'chair'), ('crib', 'crib')
        ]
        self.compress_amenities(unique_amenities_list, 'wifi', 'wifi')
        for pattern in patterns_to_compress:
            label, *patterns = pattern
            self.compress_amenities(self.filtered_amenities_list, label, *patterns)
    
        return self

    def transform(self, X):
        filtered_insight_OHE_2 = X.copy()
        filtered_insight_OHE_2['amenities'] = filtered_insight_OHE_2['amenities'].apply(ast.literal_eval)

        for value in self.filtered_amenities_list:
            appear = False
            if value in self.encode_dict.keys():
                word_filter = self.encode_dict[value]
            else:
                word_filter = [value]
            filtered_insight_OHE_2[value] = filtered_insight_OHE_2.apply(lambda row: 1.0 if self.check_amenities_in_filter(row['amenities'], word_filter) else 0.0, axis=1)

        return filtered_insight_OHE_2

    def compress_amenities(self, amenities_list, label, *patterns):
        pattern = re.compile('|'.join(patterns), re.IGNORECASE)
        var_list = [word for word in amenities_list if pattern.search(word)]
        self.filtered_amenities_list = [amenity for amenity in amenities_list if not pattern.search(amenity)]
        self.filtered_amenities_list.append(label)
        self.encode_dict[label] = var_list

    def check_amenities_in_filter(self, amenities, filter):
        return any(amenity in filter for amenity in amenities)
    def get_feature_names_out(self):
        return self.filtered_amenities_list


#---------------------------------- CLASS TO PERFORM ALL PREMODEL DATA PROCESSING ------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
class DataPreProcess:
    def __init__(self) -> None:
        pass
    
    #---------------------------------------------------------------------------------------------------------
    #----------- MAKE ALL TRANSFORMATIONS TO THE DATABASE -------------------
    #---------------------------------------------------------------------------------------------------------
    def transform_bdd_2(self,bdd):
        
        def filtes_real_features(bdd, features):
            col_names = list(bdd.columns)
            real_features = [elem for elem in features if elem in col_names]

            return np.array(real_features)
        if 'id' in bdd.columns:
            bdd = bdd.dropna(subset=['id'])
        # 3---------------------------------------DROP COLUMNS-----------------------------------------------------
        col_names = ["Unnamed: 0.1","Unnamed: 0","listing_url","scrape_id","last_scraped","source","name"
                "neighborhood_overview","picture_url","host_id","host_url","host_name","host_since",
                "host_location","host_about",
                "host_thumbnail_url","host_picture_url","host_neighbourhood","host_verifications","instant_bookable",
                "neighbourhood","neighborhood_overview","name","has_availability","calendar_last_scraped","calendar_updated"]

        columns_to_drop = [col for col in col_names if col in bdd.columns]
        bdd.drop(columns=columns_to_drop, inplace=True)
        # 4------------------------------------FORMATTING BOOLEAN CLUMNS-------------------------------------------------------
        replacement_dict = {"f": 0.0, "t": 1.0}
        columns_to_replace_aux = ["host_is_superhost","host_has_profile_pic","host_identity_verified"]
        columns_to_replace = [col for col in columns_to_replace_aux if col in bdd.columns]
        bdd[columns_to_replace] = bdd[columns_to_replace].replace(replacement_dict)
        # 5------------------------------------FORMATTING PRICE----------------------------------------------
        if "price" in bdd.columns:
            try:
                bdd["price"] = bdd["price"].apply(lambda x: float(x.replace("$", "").replace(",", "")))
            except (AttributeError, ValueError) as e:
                print("Price already transformed to float")
            
        # 6-----------------------------------FORMAT DATES--------------------------------------------------
        
        date_features = ['first_review','last_review']
        date_features = filtes_real_features(bdd,date_features)
        for col in date_features:
            try:
                bdd[col] = pd.to_datetime(bdd[col], format="%Y-%m-%d")
                bdd[col] = (bdd[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
            except (ValueError, TypeError) as e:
                try:
                    bdd[col] = pd.to_datetime(bdd[col], format="%m/%d/%Y")
                    bdd[col] = (bdd[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                except (ValueError, TypeError) as e:
                    print("Date already tranformed")

        # 7-----------------------------------FORMAT bathrooms_text--------------------------------------------------
        if 'bathrooms_text' in bdd.columns:
            bdd["bathrooms"] = bdd["bathrooms_text"].str.extract(r'(\d+(\.\d+)?)', expand=False)[0].astype(float)

        # 8---------------------------------BOOLS------------------------------------------------------------------
        if 'license' in bdd.columns:
            bdd["has_license"] = np.where(bdd["license"].notnull(), 1, 0)

        # Remove '%' and convert to float
        bdd["host_acceptance_rate_float"] = bdd["host_acceptance_rate"].str.rstrip('%').astype('float') / 100.0
        bdd["host_response_rate_float"] = bdd["host_response_rate"].str.rstrip('%').astype('float') / 100.0
        
        # 7-----------------------------------Tranformer-------------------------------------------------------------
        hot_features = ['neighbourhood_group_cleansed','neighbourhood_cleansed',"property_type","room_type","host_response_time"]
        num_features = ['host_listings_count','host_total_listings_count','host_has_profile_pic','host_identity_verified',
                        'accommodates','bedrooms','beds','price','minimum_nights','maximum_nights',
                        'minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm',
                        'maximum_nights_avg_ntm','availability_30','availability_60','availability_90','availability_365','number_of_reviews',
                        'number_of_reviews_ltm','number_of_reviews_l30d','calculated_host_listings_count','calculated_host_listings_count_entire_homes',
                        'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','reviews_per_month','first_review','last_review',"size",
                        'Eucld_Sagrada Família', 'Eucld_La Pedrera', 'Eucld_Liceu', 'Eucld_La Sagrera', 'Eucld_Radere Montjuic', 'Eucld_Plaça catalunya',
                        'Eucld_Ciutadella', 'Eucld_Sants', 'Eucld_Parc Güell', 'Eucld_Barceloneta', 'Eucld_Clínic','Eucld_Catedral BCN',
                        'Eucld_Besòs', 'Eucld_Turó de la Peira', 'Eucld_Les Corts', 'Eucld_Razzmatazz', 'Eucld_Platja nova', 'Eucld_Camp Nou',
                        'Eucld_Raval', 'Eucld_Parc N Icaria', 'bathrooms',"has_license","host_response_rate_float","host_acceptance_rate_float"]
        
        other_features = ['host_is_superhost','amenities','id','sentiment','sent1','sent2','sent3','sent4','sent5','sent6','sent7',
                          'sent8','sent9','sent10',"capacity","alowed_under_25","family_friendly"] 
        
        #this function will ensure that all features that we are going to be processing, are in the current database       
        hot_features = filtes_real_features(bdd,hot_features)
        num_features = filtes_real_features(bdd,num_features)
        other_features = filtes_real_features(bdd,other_features)
        
        if 'sentiment' not in list(bdd.columns):
            bdd = bdd.assign(sentiment=0)
        
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler',StandardScaler())        
        ])
        
        def identity(df):
            return df
        iden = FunctionTransformer(identity)
        
        pipe = ColumnTransformer([
            ('1HOT',OneHotEncoder(), hot_features),
            ('numbers',num_pipeline, num_features),
            ('identity',iden, other_features)
        ],
            sparse_threshold=0 )

        pipe.fit(bdd)
        
        bdd = pd.DataFrame(pipe.transform(bdd), columns = list(pipe.named_transformers_['1HOT'].get_feature_names_out()) + list(pipe.named_transformers_['numbers'].get_feature_names_out()) + list(other_features))
        if 'id' in bdd.columns:
            bdd = bdd.dropna(subset=['id'])
        
        if 'amenities' in bdd.columns:
            amenities_transformer = AmenitiesTransformer()
            bdd = amenities_transformer.fit_transform(bdd)
        
        bdd.fillna(0, inplace=True)
        
        col_to_drop = filtes_real_features(bdd,['id','amenities'])
        bdd.drop(columns=col_to_drop, inplace=True)
        return bdd
    
    ## --------- TEXT EMBEDDING: BERT -------------------------------------------------
    ## ---------------------------------------------- USED TO DESCRIBE THE DESCRIPTION--------------------------------------
    ## Call only if needed, we will do it once and then save the database, after the transformations, we will append the embeded description
    def calculate_embedings(self,bdd,column_to_embed,num_min_reviews = 100):
        # Load the pre-trained BERT model and tokenizer (you need to install transformers)
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        text_column = bdd[column_to_embed]

        # Initialize an empty list to store the embeddings
        embeddings = []

        # Loop through each element in the column and embed it
        for text in text_column:
            # Tokenize and convert the sentence to input format
            if type(text)==float:
                text = ""
            if text == np.nan:
                text = ""
            tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

            # Get the sentence embedding
            with torch.no_grad():
                outputs = model(**tokens)
                sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            # Append the embedding to the list
            embeddings.append(sentence_embedding)

        # Convert the list of embeddings to a NumPy array
        embeddings_array = np.array(embeddings)

        # train and fit the PCA for description embeded
        embeddings_array_t = embeddings_array

        variance = 0.95 

        emb = embeddings_array_t.copy()

        pca_description = PCA(n_components = variance)
        compressed = pca_description.fit_transform(emb)

        compressed = compressed.T

        print(f"Amound of components from using BERT embedding: {len(embeddings_array_t.T)}")
        print(f"Amound of components from using BERT embedding and using a PCA: {len(compressed)}")


        description_EMB_PCA = pd.DataFrame()
        description_EMB = pd.DataFrame()

        column_titles = [f'{column_to_embed}_{i}' for i in range(len(compressed))]
        for i in range(len(compressed)):
            description_EMB_PCA[column_titles[i]] = compressed[i]

        column_titles = [f'{column_to_embed}_{i}' for i in range(len(embeddings_array_t.T))]
        for i in range(len(embeddings_array_t.T)):
            description_EMB[column_titles[i]] = embeddings_array_t.T[i]

        
        #Save the embedings because it is costrly to run this function
        description_EMB_PCA.to_csv(f'bdd_barcelona/processed/Test-4/{column_to_embed}_{num_min_reviews}_EMB_PCA.csv', index=False)
        description_EMB.to_csv(f'bdd_barcelona/processed/Test-4/{column_to_embed}_{num_min_reviews}_EMB.csv', index=False)
        
        return (description_EMB_PCA,description_EMB)

    #------------ LOAD EMBEDDINGS PRECALCULATED ------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def load_description_embeddings(self,bdd,column_to_embed, num_min_reviews = 100):

        document_name_PCA = f'bdd_barcelona/processed/Test-4/{column_to_embed}_{num_min_reviews}_EMB_PCA.csv'
        document_name = f'bdd_barcelona/processed/Test-4/{column_to_embed}_{num_min_reviews}_EMB.csv'
        # Check if the document exists in the folder
        if os.path.exists( document_name_PCA) :
            print(f"Loading {column_to_embed} Embeddings")
            description_EMB_PCA = pd.read_csv(document_name_PCA)
            description_EMB = pd.read_csv(document_name)
            print('Loaded')

        else:
            print(f"Calculating Full {column_to_embed} Embeddings")
            description_EMB_PCA, description_EMB = self.calculate_embedings(bdd,column_to_embed)
            print("Calculated")
        try:
            columns_to_drop = [col for col in [{column_to_embed}] if col in bdd.columns] # per seguretat
            bdd.drop(columns=columns_to_drop, inplace=True)
        except:
            pass
        return (description_EMB_PCA,description_EMB)
    

    #------------- FUNCTION TO MAKE ALL PREMODEL ACTIONS (MOST IMPORTANT) --------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def bdd_premodel(self,df,directory = "bdd_barcelona/AB/normal",case = 0, recalculate = False, num_min_reviews = 100): 
        outpus_types_list = ['review_scores_rating', 'review_scores_accuracy',
            'review_scores_cleanliness', 'review_scores_checkin',
            'review_scores_communication', 'review_scores_location',
            'review_scores_value',]
        outpus_types = [col for col in outpus_types_list if col in df.columns]
        print("-----------------------------PREMODEL: START -------------------------")
        doc_name_A = f"{directory}/A.csv"
        doc_name_B = f"{directory}/B.csv"
        # Check if the document exists in the folder
        if os.path.exists(doc_name_A) & (not recalculate):
            print("-----------------------------PREMODEL: Already calculated -------------------------")
            A = pd.read_csv(doc_name_A)
            B = pd.read_csv(doc_name_B)
            print("-----------------------------PREMODEL: LOADED -------------------------")
        else:
            print("-----------------------------PREMODEL: Not found, satrt calculations -------------------------")
            A = df.drop(columns=outpus_types, axis=1)
            B = df[outpus_types].copy()
            print(f"Shape of A before transforming: {A.shape}")
            A = self.transform_bdd_2(A)
            # Cal comprovar si és millor amb o sense descriptió
            if case == 1:
                description_EMB_PCA,description_EMB = self.load_description_embeddings(df,'description',num_min_reviews)
                A = pd.concat([A, description_EMB_PCA], axis=1)
            # Cal comprovar si és millor amb o sense nei_overview
            if case == 2:
                description_EMB_PCA,description_EMB = self.load_description_embeddings(df,'neighborhood_overview',num_min_reviews)
                A = pd.concat([A, description_EMB_PCA], axis=1)
            if case == 3:
                description_EMB_PCA,description_EMB = self.load_description_embeddings(df,'description',num_min_reviews)
                A = pd.concat([A, description_EMB_PCA], axis=1)
                description_EMB_PCA,description_EMB = self.load_description_embeddings(df,'neighborhood_overview',num_min_reviews)
                A = pd.concat([A, description_EMB_PCA], axis=1)
            print(f"Shape of A after full transforming: {A.shape}")
            print("Now the database A is ready to be used")
            print("---------------------------PREMODEL: FINISH ---------------------------")
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            A.to_csv(doc_name_A, index = False)
            B.to_csv(doc_name_B, index = False)
        
        return (A,B)
    

    #-------------- SHOW A TABLE WITH ALL ATRIBUTES, ITS TYPE AND AN EXAMPLE  -------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def describe_dataframe(df):
        # Create an empty list to store the result data
        result_data = []

        # Iterate through the columns of the input DataFrame
        for column_name in df.columns:
            data_type = df[column_name].dtypes  # Get the data type of the column
            example_value = df[column_name].iloc[0]  # Get the first value in the column

            # Append the information as a dictionary to the result list
            result_data.append({'Variable': column_name, 'Data Type': data_type, 'Example Value': example_value})

        # Convert the result list to a DataFrame
        result_df = pd.DataFrame(result_data)

        # Convert the result DataFrame to an HTML table
        html_table = result_df.to_html(classes='table table-striped table-hover', escape=False, index=False)

        display(HTML(html_table))

#----------------------------------- CLASS TO TRAIN THE RF MODEL ------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
class ModelMaking:

    def __init__(self) -> None:
        self.data_processing_tool = DataPreProcess()
        self.score_features =[]
        pass
    
    #-------------------------------  SHOW RESULTS -----------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def show_pretty_r2_table(self,test,pred,min_reviews,results_df = None):
        table = PrettyTable()
        table.field_names = ["Model", "Score type", "R^2", "MSE"]
        try:
            if results_df is None:
                results_df = pd.DataFrame(columns=["Model", "Score type", "R^2", "MSE"])
        except Exception as e:
            pass
        rows_to_append = []
        total_r2_score = 0
        if isinstance(pred[0], np.ndarray):
            num_elements = len(pred[0])

            for i in range(num_elements):
                mse = mean_squared_error(test.values.T[i], pred.T[i])
                r2 = r2_score(test.values.T[i], pred.T[i])
                table.add_row([f"RF >{min_reviews}revi", test.columns[i], f"{r2:.4f}", f"{mse:.4f}"])
                rows_to_append.append({"Model": f"RF >{min_reviews}revi",
                                    "Score type": test.columns[i],
                                    "R^2": r2,
                                    "MSE": mse})
                total_r2_score += r2
        else:
            num_elements = 1

            for i in range(num_elements):
                mse = mean_squared_error(test, pred)
                r2 = r2_score(test, pred)


                #mse = mean_squared_error(test.values.T, pred.T)
                #r2 = r2_score(test.values.T, pred.T)
                table.add_row([f"RF >{min_reviews}revi", test.columns[i], f"{r2:.4f}", f"{mse:.4f}"])
                rows_to_append.append({"Model": f"RF >{min_reviews}revi",
                                    "Score type": test.columns[i],
                                    "R^2": r2,
                                    "MSE": mse})
                total_r2_score += r2
        
        print(table)
        print(total_r2_score)
        return pd.concat([results_df, pd.DataFrame(rows_to_append)], ignore_index=True, axis = 0)

    #-------------------------------- TRAIN A RF MODEL ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def model_testing(self,A,B,name=25,results_df = None,recalculate=False):
        
        directory = fr'bdd_barcelona/models/{name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"random_forest_model_{name}.pkl")
        #if results_df == None:
            #results_df = pd.DataFrame(columns=["Model", "Score type", "R^2", "MSE"])
        print("-------------- MODEL TRAINING: START ----------------------------")
        A.fillna(A.mean(), inplace=True)
        B.fillna(B.mean(), inplace=True)

        # Split the dataset into training and test sets (e.g., 80% train, 20% test)
        A.columns = A.columns.astype(str)
        A.columns = [str(col) for col in A.columns]
        B.columns = B.columns.astype(str)
        B.columns = [str(col) for col in B.columns]
        X_train, X_test, y_train, y_test = train_test_split(A, B, test_size=0.2, random_state=42)
        X_test.columns = X_test.columns.astype(str)
        X_train.columns = X_train.columns.astype(str)
        y_train.columns = y_train.columns.astype(str)
        y_test.columns = y_test.columns.astype(str)

        regressor = RandomForestRegressor(n_estimators=100,random_state=0)

        regressor.fit(X_train, y_train)
        
        # Save the trained model to a file
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(regressor, pickle_file)
            
        print("-------------- MODEL TRAINING: FINISH ----------------------------")
        y_pred_rf = regressor.predict(X_test)

        results_df = self.show_pretty_r2_table(y_test,y_pred_rf,name,results_df)
        results_df.to_csv(fr'{directory}/results_{name}.csv', index=False)
        return results_df
    
    #--------------------------------- LOAD A RF MODEL ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def load_rf_model(self,filtered_insights, name = 'latest',recalculate = False):
    
        directory = fr'bdd_barcelona/models/{name}'  
        model_path = os.path.join(directory, f'random_forest_model_{name}.pkl')      
            
        # Check if the document exists in the folder
        if os.path.exists(model_path) & (not recalculate):
            print("-----------------------------Model Found-----------------------------")
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            A = pd.read_csv(fr'{directory}/A.csv')
            B = pd.read_csv(fr'{directory}/B.csv')
            self.score_features = list(B.columns)
            results_df = pd.read_csv(fr'{directory}/results_{name}.csv')

            display(results_df)
            print('----------------------------Model Loaded-----------------------------')

        else:
            print("--------------------------Model Not Found----------------------------")
            os.makedirs(directory, exist_ok=True)
            #filtered_insights = filter_accomodation_num_review(insight_bdd, 'number_of_reviews', min_reviews)
            A,B = self.data_processing_tool.bdd_premodel(filtered_insights,f"bdd_barcelona/models/{name}",0,recalculate)
            results_df = self.model_testing(A,B,name) 
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            print("-----------------------------Model Loaded----------------------------")
            
        return (loaded_model,A,B,results_df)

    #---------------------------------- PLOT ONE SCORE -----------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def visualise_outcom(self, test, pred, show_legend = True):
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        # Scatter plot of y_test vs y_pred
        ax1 = axes[0]
        ax1.scatter(test, pred, marker='o', color='blue', alpha=0.5)
        ax1.set_xlabel("True Values (test)")
        ax1.set_ylabel("Predicted Values (pred)")
        ax1.set_title("Scatter Plot of True vs Predicted Values")
        ax1.set_ylim(3, 5)
        ax1.set_xlim(3, 5)

        #y_test_reset = test.reset_index(drop=True)
        #sorted_indices = np.argsort(y_test_reset)
        #y_test_sorted = y_test_reset.iloc[sorted_indices]
        #y_pred_sorted = pred[sorted_indices]


        # Sort the values for a smoother line plot
        sorted_indices = np.argsort(test)
        y_test_sorted = test[sorted_indices]
        y_pred_sorted = pred[sorted_indices]

        # Line plot of y_test vs y_pred
        ax2 = axes[1]
        ax2.plot(range(len(test)), y_test_sorted, label="True Values (test)", marker='o', linestyle='-')
        ax2.plot(range(len(pred)), y_pred_sorted, label="Predicted Values (pred)", marker='x', linestyle='--')
        ax2.set_xlabel("Data Points")
        ax2.set_ylabel("Values")
        ax2.set_title("Line Plot of True vs Predicted Values")
        if show_legend:
            ax2.legend(loc='upper left')

        # Adjust layout
        plt.tight_layout()

        # Show the combined figure
        plt.show()

    #----------------------------------- PLOT ALL SCORES ----------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def plot_all_results(self,test,pred,show_legend = True):
        for i in range(7):
            print(f"{self.score_features[i]}")
            self.visualise_outcom(test.values.T[i], pred.T[i],show_legend)

    #------------------------------------ PLOT ONE RESIDUAL SCORE ---------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def resldual_plots(self,y_test,y_pred,index):
        # Calculate the residuals
        residuals = y_test - y_pred

        # Create a figure with two subplots: residual error plot and distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Residual Error Plot
        ax1.scatter(x=y_pred, y=residuals, color='b', marker='o', alpha=0.4, label="y_pred")
        ax1.scatter(x=y_test, y=residuals, color='g', marker='o', alpha=0.4, label = "y_test")
        ax1.set_title("Residual Error Plot")
        ax1.set_xlabel("y Values")
        ax1.set_ylabel("Residuals")
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.legend()

        # Distribution Plot
        sns.histplot(y_pred.T[index], kde=False, ax=ax2,alpha=0.5, color='g',label='y_pred')
        sns.histplot(y_test.T[index], kde=False, ax=ax2,alpha=0.5, color='b', label='y_test')
        ax2.set_title("Residuals Distribution")
        ax2.set_xlabel("Residuals")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    #------------------------------------- PLOT MODEL FEATURE IMPORTANCE --------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def feature_importance(self,X_train,model,show_full_importnce_graph = True):
        data = {'Name': X_train.columns, 'Score': model.feature_importances_}
        df = pd.DataFrame(data)
        df = df.sort_values(by='Score', ascending=False)
        top_10 = df.head(10)

        # Plot a bar chart for the top 10 features
        plt.figure(figsize=(4, 3))
        plt.barh(top_10['Name'], top_10['Score'])
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Feature Name')
        plt.title('Random Forest Top 10 Most Important Features')
        plt.gca().invert_yaxis()  # Invert the y-axis to display the most important feature at the top
        plt.show()

        if show_full_importnce_graph:
            x = [i for i in range(len(df['Score']))]
            colors = ['grey' if score < 0.009 else 'blue' for score in df['Score']]
            plt.figure(figsize=(4, 3))
            plt.scatter(x, df['Score'], c=colors, marker='o')
            plt.ylabel('Feature Importance Score')
            plt.title('Feature Importance')
            plt.xticks( rotation=90)  # Set x-axis labels and rotate for better readability
            plt.show()


