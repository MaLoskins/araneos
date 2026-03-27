# FeatureSpaceCreator.py
import os,re,pandas as pd,numpy as np,torch,torch.nn as nn
from typing import Dict,Any,List,Optional,Union
import urllib.request,zipfile,tempfile,shutil
from gensim.models import Word2Vec
from transformers import BertModel,BertTokenizerFast
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from math import ceil
from umap import UMAP
import warnings,logging,spacy

# Custom GloVe implementation to replace torchtext
class GloVe:
    def __init__(self,name="6B",dim=300,cache=None):
        """Custom GloVe implementation to replace torchtext.vocab.GloVe"""
        self.name=name;self.dim=dim;self.cache=os.path.normpath(cache) if cache else None
        self.stoi={};self.vectors=None;self.itos=[]
        
        # Create cache dir if needed
        self.cache and not os.path.exists(self.cache) and os.makedirs(self.cache,exist_ok=True)
        
        # Load vectors
        self._load_vectors()
    
    def _download_glove_vectors(self):
        """Download GloVe vectors from Stanford NLP website"""
        if not self.cache:
            raise ValueError("cache directory must be provided for GloVe embeddings")
            
        # Normalize path to handle Windows backslashes correctly
        self.cache = os.path.normpath(self.cache)
        
        # Map of available GloVe vectors
        glove_urls = {
            "6B": {
                50: "https://nlp.stanford.edu/data/glove.6B.zip",
                100: "https://nlp.stanford.edu/data/glove.6B.zip",
                200: "https://nlp.stanford.edu/data/glove.6B.zip",
                300: "https://nlp.stanford.edu/data/glove.6B.zip"
            },
            "42B": {
                300: "https://nlp.stanford.edu/data/glove.42B.300d.zip"
            },
            "840B": {
                300: "https://nlp.stanford.edu/data/glove.840B.300d.zip"
            },
            "twitter.27B": {
                25: "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
                50: "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
                100: "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
                200: "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
            }
        }
        
        # Check if the requested vectors are available
        if self.name not in glove_urls:
            raise ValueError(f"GloVe vectors for '{self.name}' are not available for download")
        
        if self.dim not in glove_urls[self.name]:
            raise ValueError(f"GloVe vectors for '{self.name}' with dimension {self.dim} are not available for download")
        
        url = glove_urls[self.name][self.dim]
        logging.info(f"Downloading GloVe vectors from {url}")
        
        # Create a temporary directory for the download
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "glove.zip")
            
            # Download the zip file
            logging.info("Downloading GloVe vectors... This may take a while.")
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            logging.info("Extracting GloVe vectors...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the correct file in the extracted directory
            vector_filename = f"glove.{self.name}.{self.dim}d.txt"
            vector_filepath = os.path.join(temp_dir, vector_filename)
            
            # For Twitter vectors, the filename format is different
            if self.name == "twitter.27B":
                vector_filename = f"glove.twitter.27B.{self.dim}d.txt"
                vector_filepath = os.path.join(temp_dir, vector_filename)
            
            # Copy the file to the cache directory
            if os.path.exists(vector_filepath):
                shutil.copy(vector_filepath, os.path.join(self.cache, vector_filename))
                logging.info(f"GloVe vectors saved to {os.path.join(self.cache, vector_filename)}")
            else:
                # Try to find any glove file in the extracted directory
                extracted_files = os.listdir(temp_dir)
                glove_files = [f for f in extracted_files if f.startswith('glove.') and f.endswith('.txt')]
                
                if glove_files:
                    for glove_file in glove_files:
                        # Check if this file matches our dimension
                        if f"{self.dim}d" in glove_file:
                            shutil.copy(os.path.join(temp_dir, glove_file), os.path.join(self.cache, glove_file))
                            logging.info(f"GloVe vectors saved to {os.path.join(self.cache, glove_file)}")
                            return
                    
                    # If no file with matching dimension is found, copy the first one
                    shutil.copy(os.path.join(temp_dir, glove_files[0]), os.path.join(self.cache, glove_files[0]))
                    logging.info(f"GloVe vectors saved to {os.path.join(self.cache, glove_files[0])}")
                else:
                    raise FileNotFoundError(f"Could not find GloVe vectors in the downloaded archive")
    
    def _load_vectors(self):
        """Load GloVe vectors from file"""
        if not self.cache:raise ValueError("cache directory must be provided for GloVe embeddings")
        
        # Normalize path
        self.cache=os.path.normpath(self.cache)
        
        # Try different filename patterns
        filepaths=[
            os.path.join(self.cache,f"glove.{self.name}.{self.dim}d.txt"),
            os.path.join(self.cache,f"glove.{self.name}.{self.dim}d.vec"),
            os.path.join(self.cache,f"glove.{self.name}.txt"),
            os.path.join(self.cache,f"glove.twitter.27B.{self.dim}d.txt") if self.name=="twitter.27B" else None
        ]
        
        # Find first existing file or any glove file
        filepath=next((p for p in filepaths if p and os.path.exists(p)),None)
        if not filepath:
            # Try any glove file
            glove_files=[f for f in os.listdir(self.cache) if f.startswith('glove.')]
            filepath=os.path.join(self.cache,glove_files[0]) if glove_files else None
            
            # Download if nothing found
            if not filepath:
                logging.info(f"No GloVe vectors found in {self.cache}. Attempting to download...")
                try:
                    self._download_glove_vectors()
                    return self._load_vectors()
                except Exception as e:
                    raise FileNotFoundError(f"No GloVe vectors found in {self.cache} and download failed: {e}")
        
        logging.info(f"Loading GloVe vectors from {filepath}")
        
        # Count lines & initialize
        with open(filepath,'r',encoding='utf-8') as f:
            num_lines=sum(1 for _ in f)
        self.vectors=np.zeros((num_lines,self.dim),dtype=np.float32)
        
        # Load vectors
        with open(filepath,'r',encoding='utf-8') as f:
            for i,line in enumerate(f):
                try:
                    values=line.rstrip().split(' ')
                    word,vector=values[0],np.array([float(v) for v in values[1:]],dtype=np.float32)
                    if len(vector)!=self.dim:continue
                    self.stoi[word]=i;self.itos.append(word);self.vectors[i]=vector
                except Exception as e:
                    logging.warning(f"Error processing line {i}: {e}")
        
        logging.info(f"Loaded {len(self.stoi)} GloVe vectors of dimension {self.dim}")
    
    def __getitem__(self,token):
        """Get embedding vector for token"""
        return torch.tensor(self.vectors[self.stoi[token]],dtype=torch.float) if token in self.stoi else torch.zeros(self.dim,dtype=torch.float)

# Removed torchtext deprecation warning
warnings.filterwarnings("ignore")

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TextPreprocessor:
    def __init__(self,target_column:str='text',include_stopwords:bool=True,remove_ats:bool=True,
                word_limit:int=100,tokenizer:Optional[Any]=None):
        self.target_column=target_column;self.include_stopwords=include_stopwords
        self.remove_ats=remove_ats;self.word_limit=word_limit
        self.tokenizer=tokenizer if tokenizer else self.spacy_tokenizer

        if include_stopwords:
            try:
                self.nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])
                self.stop_words=self.nlp.Defaults.stop_words
            except OSError:
                try:
                    import subprocess
                    subprocess.run(["python","-m","spacy","download","en_core_web_sm"],check=True)
                    self.nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])
                    self.stop_words=self.nlp.Defaults.stop_words
                except Exception as e:
                    raise OSError(f"Failed to install spaCy model 'en_core_web_sm': {e}")
        else:self.stop_words=set()

        self.re_pattern=re.compile(r'[^\w\s]');self.at_pattern=re.compile(r'@\S+')

    def spacy_tokenizer(self,text:str)->List[str]:
        return [t.text for t in self.nlp(text)]

    def clean_text(self,df:pd.DataFrame)->pd.DataFrame:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' does not exist.")

        df=df.copy()
        df[self.target_column]=df[self.target_column].astype(str).str.lower()
        
        # Apply regex replacements
        if self.remove_ats:
            df[self.target_column]=df[self.target_column].str.replace(self.at_pattern,'',regex=True)
        df[self.target_column]=df[self.target_column].str.replace(self.re_pattern,'',regex=True)
        
        # Tokenize
        df['tokenized_text']=df[self.target_column].apply(self.tokenizer)
        
        # Filter tokens
        filter_fn=lambda tokens:[w for w in tokens if (not self.include_stopwords or w not in self.stop_words) and len(w)<=self.word_limit]
        df['tokenized_text']=df['tokenized_text'].apply(filter_fn)

        return df


class EmbeddingCreator:
    def __init__(
        self,
        embedding_method: str = "bert",
        embedding_dim: int = 768,
        glove_cache_path: Optional[str] = None,
        word2vec_model_path: Optional[str] = None,
        bert_model_name: str = "bert-base-uncased",
        bert_cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        self.embedding_method = embedding_method.lower()
        self.embedding_dim = embedding_dim
        self.glove = None
        self.word2vec_model = None
        self.bert_model = None
        self.tokenizer = None

        self.device = torch.device("cpu")
        if device == "cuda" and torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                self.device = torch.device("cuda")
            except Exception:
                logging.warning("CUDA available but unusable (kernel mismatch), falling back to CPU")

        if self.embedding_method == "glove":
            self._load_glove(glove_cache_path)
        elif self.embedding_method == "word2vec":
            if word2vec_model_path:
                self._load_word2vec(word2vec_model_path)
            else:
                self.word2vec_model = None  # To be trained later
        elif self.embedding_method == "bert":
            self._load_bert(bert_model_name, bert_cache_dir)
        else:
            raise ValueError("Unsupported embedding method. Choose from 'glove', 'word2vec', or 'bert'.")

    def _load_glove(self, glove_cache_path: str):
        if not glove_cache_path:
            raise ValueError("glove_cache_path must be provided for GloVe embeddings.")
        if not os.path.exists(glove_cache_path):
            raise FileNotFoundError(f"GloVe cache path '{glove_cache_path}' does not exist.")
        self.glove = GloVe(name="6B", dim=self.embedding_dim, cache=glove_cache_path)

    def _load_word2vec(self, word2vec_model_path: str):
        if not word2vec_model_path or not os.path.exists(word2vec_model_path):
            raise ValueError("A valid word2vec_model_path must be provided for Word2Vec embeddings.")
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        if self.word2vec_model.vector_size != self.embedding_dim:
            raise ValueError(f"Word2Vec model dimension ({self.word2vec_model.vector_size}) "
                             f"does not match embedding_dim ({self.embedding_dim}).")

    def _load_bert(self, bert_model_name: str, bert_cache_dir: Optional[str]):
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model = BertModel.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        self.embedding_dim = self.bert_model.config.hidden_size

    def train_word2vec(self, sentences: List[List[str]], vector_size: int = 300,
                       window: int = 5, min_count: int = 1, workers: int = 4):
        if self.embedding_method != "word2vec":
            raise ValueError("train_word2vec can only be called for 'word2vec' embedding method.")
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=42
        )

    def get_embedding(self, tokens: List[str]) -> np.ndarray:
        if self.embedding_method in ["glove", "word2vec"]:
            if self.embedding_method == "word2vec" and self.word2vec_model is None:
                raise ValueError("Word2Vec model is not trained. Please train the model before getting embeddings.")
            return self._get_average_embedding(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_embedding(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def get_word_embeddings(self, tokens: List[str]) -> np.ndarray:
        if self.embedding_method in ["glove", "word2vec"]:
            if self.embedding_method == "word2vec" and self.word2vec_model is None:
                raise ValueError("Word2Vec model is not trained. Please train the model before getting embeddings.")
            return self._get_individual_embeddings(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_word_embeddings(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def _get_average_embedding(self,tokens:List[str])->np.ndarray:
        # Get valid embeddings with list comprehension
        embs=[
            self.glove[t].numpy() if self.embedding_method=="glove" and t in self.glove.stoi else
            self.word2vec_model.wv[t] if self.embedding_method=="word2vec" and t in self.word2vec_model.wv else
            None for t in tokens
        ]
        # Filter None values and compute mean
        valid_embs=[e for e in embs if e is not None]
        return np.mean(valid_embs,axis=0) if valid_embs else np.zeros(self.embedding_dim)

    def _get_individual_embeddings(self,tokens:List[str])->np.ndarray:
        # One-liner with ternary operators
        return np.array([
            self.glove[t].numpy() if self.embedding_method=="glove" and t in self.glove.stoi else
            self.word2vec_model.wv[t] if self.embedding_method=="word2vec" and t in self.word2vec_model.wv else
            np.zeros(self.embedding_dim) for t in tokens
        ])

    def _get_bert_embedding(self, tokens: List[str]) -> np.ndarray:
        text = ' '.join(tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    def _get_bert_embedding_batch(self, list_of_token_lists: List[List[str]]) -> np.ndarray:
        """
        Batch process multiple tokenized rows with BERT, returning
        an array of shape (batch_size, hidden_size).
        Each row's embedding is the [CLS] token from BERT.
        """
        texts = [' '.join(tokens) for tokens in list_of_token_lists]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embeddings

    def _get_bert_word_embeddings(self, tokens: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state.squeeze(0)
            word_ids = inputs.word_ids(batch_index=0)
            if word_ids is None:
                raise ValueError("word_ids() returned None. Ensure you are using a fast tokenizer.")
            word_embeddings = {}
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id not in word_embeddings:
                        word_embeddings[word_id] = []
                    word_embeddings[word_id].append(last_hidden_state[idx].cpu().numpy())
            averaged_embeddings = []
            for wid in sorted(word_embeddings.keys()):
                arr = np.array(word_embeddings[wid])
                averaged_embeddings.append(arr.mean(axis=0))
        return np.array(averaged_embeddings)


class FeatureSpaceCreator:
    def __init__(self, config: Dict[str, Any], device: str = "cuda", log_file: str = "logs/feature_space_creator.log"):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.features = config.get("features", [])
        self.multi_graph_settings = config.get("multi_graph_settings", {})

        self.text_features = []
        self.numeric_features = []

        self.logger = self._setup_logger(log_file=log_file)
        self._parse_config()

        self.embedding_creators = {}
        for feature in self.text_features:
            method = feature.get("embedding_method", "bert").lower()
            embedding_dim = feature.get("embedding_dim", None)
            additional_params = feature.get("additional_params", {})

            try:
                self.embedding_creators[feature["column_name"]] = EmbeddingCreator(
                    embedding_method=method,
                    embedding_dim=embedding_dim,
                    glove_cache_path=additional_params.get("glove_cache_path"),
                    word2vec_model_path=additional_params.get("word2vec_model_path"),
                    bert_model_name=additional_params.get("bert_model_name", "bert-base-uncased"),
                    bert_cache_dir=additional_params.get("bert_cache_dir"),
                    device=self.device
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize EmbeddingCreator for '{feature['column_name']}': {e}")
                raise e

        self.scalers = {}
        for feature in self.numeric_features:
            processing = feature.get("processing", "none").lower()
            if processing == "standardize":
                self.scalers[feature["column_name"]] = StandardScaler()
            elif processing == "normalize":
                self.scalers[feature["column_name"]] = MinMaxScaler()

        self.projection_layers = {}
        for feature in self.numeric_features:
            projection_config = feature.get("projection", {})
            method = projection_config.get("method", "none").lower()
            target_dim = projection_config.get("target_dim", 1)

            if method == "linear" and target_dim > 1:
                projection = nn.Linear(1, target_dim).to(self.device)
                projection.eval()
                self.projection_layers[feature["column_name"]] = projection

        self.text_preprocessor = TextPreprocessor(
            target_column=None,
            include_stopwords=True,
            remove_ats=True,
            word_limit=100,
            tokenizer=None
        )

    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("FeatureSpaceCreator")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _parse_config(self):
        for feature in self.features:
            f_type = feature.get("type", "").lower()
            if f_type == "text":
                self.text_features.append(feature)
            elif f_type == "numeric":
                self.numeric_features.append(feature)
            else:
                raise ValueError(f"Unsupported feature type: '{f_type}' in feature '{feature.get('column_name')}'.")

    def process(self,dataframe:Union[str,pd.DataFrame])->pd.DataFrame:
        # Load data
        df=pd.read_csv(dataframe) if isinstance(dataframe,str) and os.path.exists(dataframe) else \
           dataframe.copy() if isinstance(dataframe,pd.DataFrame) else \
           exec("raise TypeError(\"dataframe must be a file path (str) or a pandas DataFrame.\")")
        
        self.logger.info(f"Loaded data from {'file' if isinstance(dataframe,str) else 'pandas DataFrame'}.")
        
        # Initialize feature space
        feature_space=pd.DataFrame(index=df.index)
        self.logger.info("Initialized feature space DataFrame.")

        # Process text features
        for feature in self.text_features:
            col = feature["column_name"]
            if col not in df.columns:
                raise ValueError(f"Text column '{col}' not found in the DataFrame.")

            if df[col].isnull().any():
                self.logger.warning(f"Missing values found in text column '{col}'. Filling with empty strings.")
                df[col] = df[col].fillna("")

            if self.text_preprocessor.target_column != col:
                self.text_preprocessor.target_column = col
            processed_df = self.text_preprocessor.clean_text(df)
            tokens = processed_df["tokenized_text"].tolist()

            # If Word2Vec without a pre-trained model path, we train on the fly
            if feature["embedding_method"].lower() == "word2vec":
                word2vec_model_path = feature.get("additional_params", {}).get("word2vec_model_path", None)
                if not word2vec_model_path:
                    self.logger.info(f"Training Word2Vec model for '{col}' as no model path was provided.")
                    self.embedding_creators[col].train_word2vec(sentences=tokens)
                    self.logger.info(f"Word2Vec model trained for '{col}'.")

            # BERT BATCHING:
            method = feature["embedding_method"].lower()
            bert_batch_size = 1
            if method == "bert":
                bert_batch_size = feature.get("additional_params", {}).get("bert_batch_size", 1)

            if method == "bert" and bert_batch_size > 1:
                self.logger.info(f"Using batched BERT embeddings for '{col}' with batch size={bert_batch_size}.")
                embeddings = []
                total_rows = len(tokens)
                # chunk tokens by batch_size
                for i in range(0, total_rows, bert_batch_size):
                    batch_chunk = tokens[i : i + bert_batch_size]
                    embeddings_chunk = self.embedding_creators[col]._get_bert_embedding_batch(batch_chunk)
                    embeddings.append(embeddings_chunk)
                embeddings_array = np.vstack(embeddings)
            else:
                # Vectorized embedding with list comprehension
                embeddings_array=np.vstack([self.embedding_creators[col].get_embedding(t) for t in tokens])

            self.logger.info(f"Generated embeddings for text column '{col}' with shape {embeddings_array.shape}.")

            dim_reduction_config = feature.get("dim_reduction", {})
            method = dim_reduction_config.get("method", "none").lower()
            target_dim = dim_reduction_config.get("target_dim", embeddings_array.shape[1])

            if method in ["pca", "umap"] and target_dim < embeddings_array.shape[1]:
                self.logger.info(f"Applying '{method}' to text feature '{col}' to reduce dimensions to {target_dim}.")
                if method == "pca":
                    reducer = PCA(n_components=target_dim, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embeddings_array)
                elif method == "umap":
                    n_neighbors = dim_reduction_config.get("n_neighbors", 15)
                    min_dist = dim_reduction_config.get("min_dist", 0.1)
                    reducer = UMAP(n_components=target_dim, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embeddings_array)

                embeddings_array = reduced_embeddings
                self.logger.info(f"Dimensionality reduction '{method}' applied to '{col}'. "
                                 f"New shape: {embeddings_array.shape}.")

            feature_space[f"{col}_embedding"] = list(embeddings_array)

        # Process numeric features
        for feature in self.numeric_features:
            col = feature["column_name"]
            if col not in df.columns:
                raise ValueError(f"Numeric column '{col}' not found in the DataFrame.")

            if df[col].isnull().any():
                self.logger.warning(f"Missing values found in numeric column '{col}'. Filling with column mean.")
                df[col] = df[col].fillna(df[col].mean())

            data_type = feature.get("data_type", "float").lower()
            if data_type not in ["int", "float"]:
                raise ValueError(f"Unsupported data_type '{data_type}' for numeric column '{col}'.")

            df[col] = df[col].astype(float) if data_type == "float" else df[col].astype(int)

            processing = feature.get("processing", "none").lower()
            if processing in ["standardize", "normalize"]:
                scaler = self.scalers[col]
                df_scaled = scaler.fit_transform(df[[col]])
                feature_vector = df_scaled.flatten()
                self.logger.info(f"Applied '{processing}' to numeric column '{col}'.")
            else:
                feature_vector = df[col].values.astype(float)
                self.logger.info(f"No scaling applied to numeric column '{col}'.")

            projection_config = feature.get("projection", {})
            method = projection_config.get("method", "none").lower()
            target_dim = projection_config.get("target_dim", 1)

            if method == "linear" and target_dim > 1:
                self.logger.info(f"Applying '{method}' projection to numeric feature '{col}' to increase dimensions to {target_dim}.")
                projection_layer = self.projection_layers[col]
                with torch.no_grad():
                    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(1).to(self.device)
                    projected_tensor = projection_layer(feature_tensor)
                    projected_features = projected_tensor.cpu().numpy()
                feature_space[f"{col}_feature"] = list(projected_features)
                self.logger.info(f"Projection '{method}' applied to '{col}'. New shape: {projected_features.shape}.")
            else:
                feature_space[f"{col}_feature"] = feature_vector
                self.logger.info(f"Added numeric feature '{col}' with shape {feature_vector.shape}.")

        self.logger.info("Feature space creation completed.")
        return feature_space