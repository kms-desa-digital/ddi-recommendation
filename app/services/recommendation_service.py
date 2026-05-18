import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from ..config.mongodb_config import db
from ..utils.text_processing import (
    preprocess_text,
    preprocess_combined_features,
    batch_preprocess_text,
)
from ..models.recommendation import RecommendationResponse


class RecommendationEngine:
    """
    Optimized recommendation engine dengan caching dan error handling yang lebih baik
    """

    def __init__(self, cache_ttl: int = 3600):  # Cache TTL dalam detik (default: 1 jam)
        self.df: Optional[pd.DataFrame] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.cosine_sim: Optional[np.ndarray] = None
        self.cache_ttl = cache_ttl
        self.last_update = 0
        self.lock = threading.Lock()
        self.is_initialized = False
        self.initialize_model()

    def _fetch_innovations_from_mongodb(self) -> List[Dict[str, Any]]:
        """
        Fetch innovations dari MongoDB dengan error handling

        Returns:
            List[Dict[str, Any]]: List data inovasi
        """
        try:
            if db is None:
                print("MongoDB connection is not established.")
                return []
                
            # Query innovations, prioritize verified ones
            cursor = db.innovations.find({"status": "Terverifikasi"})
            data = []

            for doc in cursor:
                try:
                    doc_data = {}
                    # Map MongoDB fields
                    doc_data["id"] = str(doc.get("_id"))
                    doc_data["deskripsi"] = doc.get("deskripsi", "")
                    doc_data["kategori"] = doc.get("kategori", "")
                    doc_data["namaInovasi"] = doc.get("namaInovasi", "")
                    doc_data["namaInnovator"] = doc.get("namaInnovator", "")
                    
                    # Handle images/fotoInovasi field
                    raw_images = doc.get("fotoInovasi") or doc.get("images") or []
                    if isinstance(raw_images, str):
                        doc_data["images"] = [raw_images]
                    elif isinstance(raw_images, list):
                        doc_data["images"] = raw_images
                    else:
                        doc_data["images"] = []
                        
                    doc_data["tahunDibuat"] = str(doc.get("tahunDibuat", ""))
                    
                    data.append(doc_data)
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue

            # Fallback if no verified innovations are found (e.g. initial setup)
            if not data:
                print("No verified innovations found, querying all innovations as fallback...")
                cursor_all = db.innovations.find()
                for doc in cursor_all:
                    try:
                        doc_data = {}
                        doc_data["id"] = str(doc.get("_id"))
                        doc_data["deskripsi"] = doc.get("deskripsi", "")
                        doc_data["kategori"] = doc.get("kategori", "")
                        doc_data["namaInovasi"] = doc.get("namaInovasi", "")
                        doc_data["namaInnovator"] = doc.get("namaInnovator", "")
                        
                        raw_images = doc.get("fotoInovasi") or doc.get("images") or []
                        if isinstance(raw_images, str):
                            doc_data["images"] = [raw_images]
                        elif isinstance(raw_images, list):
                            doc_data["images"] = raw_images
                        else:
                            doc_data["images"] = []
                            
                        doc_data["tahunDibuat"] = str(doc.get("tahunDibuat", ""))
                        data.append(doc_data)
                    except Exception as e:
                        print(f"Error processing document: {e}")
                        continue

            print(f"Fetched {len(data)} innovations from MongoDB.")
            return data
        except Exception as e:
            print(f"Error fetching innovations from MongoDB: {e}")
            return []

    def _validate_and_clean_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Validasi dan pembersihan data yang diambil dari Firebase

        Args:
            data (List[Dict[str, Any]]): Raw data dari Firebase

        Returns:
            pd.DataFrame: Clean DataFrame
        """
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Validasi kolom yang diperlukan
        required_columns = [
            "id",
            "deskripsi",
            "kategori",
            "namaInovasi",
            "namaInnovator",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in data: {missing_columns}")
            # Tambahkan kolom yang hilang dengan nilai default
            for col in missing_columns:
                df[col] = ""

        # Bersihkan data yang tidak valid
        df = df.dropna(subset=["id"])  # ID harus ada
        df = df[df["id"].str.strip() != ""]  # ID tidak boleh kosong

        # Handle missing values untuk kolom teks
        text_columns = ["deskripsi", "kategori", "namaInovasi", "namaInnovator"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        return df

    def _create_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Membuat combined features dengan optimasi

        Args:
            df (pd.DataFrame): DataFrame input

        Returns:
            pd.DataFrame: DataFrame dengan combined features
        """
        # Metode 1: Simple concatenation (default)
        df["combined_features"] = df["deskripsi"] + " " + df["kategori"]

        # Metode 2: Weighted category (commented out, bisa diaktifkan jika diperlukan)
        # df['combined_features'] = df.apply(
        #     lambda row: preprocess_combined_features(
        #         row['deskripsi'], row['kategori'], kategori_weight=2
        #     ), axis=1
        # )

        return df

    def _batch_preprocess_texts(self, texts: pd.Series) -> List[str]:
        """
        Batch preprocessing untuk performa yang lebih baik

        Args:
            texts (pd.Series): Series teks yang akan diproses

        Returns:
            List[str]: List teks yang sudah diproses
        """
        return batch_preprocess_text(texts.tolist())

    def _build_tfidf_matrix(self, processed_texts: List[str]) -> tuple:
        """
        Membangun TF-IDF matrix dengan parameter yang dioptimasi

        Args:
            processed_texts (List[str]): List teks yang sudah diproses

        Returns:
            tuple: (TfidfVectorizer, tfidf_matrix)
        """
        # Filter teks kosong
        valid_texts = [text for text in processed_texts if text.strip()]

        if len(valid_texts) < 2:
            print("Warning: Insufficient valid texts for TF-IDF")
            return None, None

        # Konfigurasi TF-IDF dengan parameter yang dioptimasi
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),  # Unigram dan bigram
            min_df=2,  # Minimum document frequency
            max_df=0.85,  # Maximum document frequency (turun dari 0.9)
            max_features=5000,  # Batasi jumlah fitur untuk performa
            stop_words=None,  # Stopwords sudah dihapus di preprocessing
            lowercase=False,  # Sudah lowercase di preprocessing
            token_pattern=r"\b\w+\b",  # Pattern untuk tokenisasi
        )

        try:
            tfidf_matrix = tfidf.fit_transform(processed_texts)
            return tfidf, tfidf_matrix
        except Exception as e:
            print(f"Error building TF-IDF matrix: {e}")
            return None, None

    def _calculate_similarity_matrix(self, tfidf_matrix) -> Optional[np.ndarray]:
        """
        Menghitung cosine similarity matrix dengan optimasi memori

        Args:
            tfidf_matrix: TF-IDF matrix

        Returns:
            Optional[np.ndarray]: Cosine similarity matrix
        """
        try:
            # Untuk dataset besar, pertimbangkan menggunakan sparse matrix
            if tfidf_matrix.shape[0] > 1000:
                # Gunakan batch processing untuk dataset besar
                return cosine_similarity(tfidf_matrix, dense_output=False).toarray()
            else:
                return cosine_similarity(tfidf_matrix, tfidf_matrix)
        except Exception as e:
            print(f"Error calculating similarity matrix: {e}")
            return None

    def initialize_model(self, force_refresh: bool = False) -> bool:
        """
        Inisialisasi model dengan caching dan error handling

        Args:
            force_refresh (bool): Paksa refresh cache

        Returns:
            bool: True jika inisialisasi berhasil
        """
        current_time = time.time()

        # Cek apakah perlu refresh cache
        if (
            not force_refresh
            and self.is_initialized
            and (current_time - self.last_update) < self.cache_ttl
        ):
            return True

        with self.lock:
            try:
                print("Initializing recommendation model...")

                # 1. Fetch data dari MongoDB
                data = self._fetch_innovations_from_mongodb()
                if not data:
                    print("No data fetched from MongoDB")
                    return False
                # 2. Validasi dan pembersihan data
                df = self._validate_and_clean_data(data)
                if df.empty:
                    print("No valid data after cleaning")
                    return False

                print(f"Processing {len(df)} innovations...")

                # 3. Buat combined features
                df = self._create_combined_features(df)

                # 4. Batch preprocessing
                processed_texts = self._batch_preprocess_texts(df["combined_features"])

                # Filter baris dengan teks kosong
                valid_indices = [
                    i for i, text in enumerate(processed_texts) if text.strip()
                ]

                if len(valid_indices) < 2:
                    print("Insufficient valid processed texts")
                    return False

                # Update dataframe dan processed_texts
                df = df.iloc[valid_indices].reset_index(drop=True)
                processed_texts = [processed_texts[i] for i in valid_indices]
                df["processed_text"] = processed_texts

                # 5. Build TF-IDF matrix
                tfidf, tfidf_matrix = self._build_tfidf_matrix(processed_texts)
                if tfidf is None or tfidf_matrix is None:
                    print("Failed to build TF-IDF matrix")
                    return False

                # 6. Calculate similarity matrix
                cosine_sim = self._calculate_similarity_matrix(tfidf_matrix)
                if cosine_sim is None:
                    print("Failed to calculate similarity matrix")
                    return False

                # 7. Update instance variables
                self.df = df
                self.tfidf = tfidf
                self.cosine_sim = cosine_sim
                self.last_update = current_time
                self.is_initialized = True

                print(f"Model initialized successfully with {len(df)} innovations")
                return True

            except Exception as e:
                print(f"Error initializing model: {e}")
                return False

    def _safe_get_value(
        self, row: pd.Series, column: str, default_value: Any = None
    ) -> Any:
        """
        Safely get value dari pandas Series dengan default handling

        Args:
            row (pd.Series): Row data
            column (str): Column name
            default_value (Any): Default value jika tidak ada/nan

        Returns:
            Any: Value atau default value
        """
        try:
            value = row.get(column, default_value)
            if pd.isna(value):
                return default_value
            return value
        except Exception:
            return default_value

    def _process_recommendation_row(
        self, row: pd.Series, score: float
    ) -> Optional[RecommendationResponse]:
        """
        Process single recommendation row dengan error handling

        Args:
            row (pd.Series): Row data
            score (float): Similarity score

        Returns:
            Optional[RecommendationResponse]: Recommendation object atau None jika error
        """
        try:
            # Handle images dengan safe processing
            raw_images = self._safe_get_value(row, "images", [])
            if isinstance(raw_images, list):
                images = raw_images
            else:
                images = []

            # Handle tahun dengan safe processing
            raw_year = self._safe_get_value(row, "tahunDibuat", None)
            if raw_year is not None and not pd.isna(raw_year):
                tahun = str(raw_year)
            else:
                tahun = None

            # Create recommendation object
            rec = RecommendationResponse(
                id=str(self._safe_get_value(row, "id", "")),
                inovasi=str(self._safe_get_value(row, "namaInovasi", "")),
                kategori=str(self._safe_get_value(row, "kategori", "")),
                deskripsi=str(self._safe_get_value(row, "deskripsi", "")),
                namaInnovator=str(self._safe_get_value(row, "namaInnovator", "")),
                images=images,
                tahunDibuat=tahun,
                similarity_score=round(score, 4),  # 4 decimal places untuk precision
            )

            return rec

        except Exception as e:
            print(f"Error processing recommendation row: {e}")
            return None

    def get_recommendations(
        self, innovation_id: str, top_n: int = 5, min_similarity: float = 0.01
    ) -> List[RecommendationResponse]:
        """
        Mendapatkan rekomendasi dengan optimasi dan error handling

        Args:
            innovation_id (str): ID inovasi yang akan dicari rekomendasinya
            top_n (int): Jumlah rekomendasi yang diinginkan
            min_similarity (float): Minimum similarity score

        Returns:
            List[RecommendationResponse]: List rekomendasi
        """
        # Validasi input
        if not innovation_id or not innovation_id.strip():
            return []

        # Cek apakah model sudah diinisialisasi
        if not self.is_initialized:
            print("Model not initialized, attempting to initialize...")
            if not self.initialize_model():
                return []

        # Auto-refresh jika cache expired
        current_time = time.time()
        if (current_time - self.last_update) > self.cache_ttl:
            print("Cache expired, refreshing model...")
            if not self.initialize_model():
                return []

        try:
            # Cari index dari innovation_id
            matching_rows = self.df[self.df["id"] == innovation_id]
            if matching_rows.empty:
                print(f"Innovation ID {innovation_id} not found")
                return []

            idx = matching_rows.index[0]

            # Dapatkan similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            # Filter berdasarkan minimum similarity
            sim_scores = [
                (i, score) for i, score in sim_scores if score >= min_similarity
            ]

            # Sort berdasarkan similarity score
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Exclude item yang sama dan ambil top_n
            sim_scores = sim_scores[1 : top_n + 1]

            # Process recommendations dengan parallel processing untuk dataset besar
            recommendations = []

            if len(sim_scores) > 10:  # Gunakan parallel processing untuk dataset besar
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for i, score in sim_scores:
                        row = self.df.iloc[i]
                        future = executor.submit(
                            self._process_recommendation_row, row, score
                        )
                        futures.append(future)

                    for future in futures:
                        try:
                            rec = future.result(timeout=5)  # 5 second timeout
                            if rec:
                                recommendations.append(rec)
                        except Exception as e:
                            print(f"Error processing recommendation: {e}")
                            continue
            else:
                # Serial processing untuk dataset kecil
                for i, score in sim_scores:
                    row = self.df.iloc[i]
                    rec = self._process_recommendation_row(row, score)
                    if rec:
                        recommendations.append(rec)

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Mendapatkan statistik model untuk monitoring

        Returns:
            Dict[str, Any]: Model statistics
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "total_innovations": len(self.df) if self.df is not None else 0,
            "last_update": self.last_update,
            "cache_ttl": self.cache_ttl,
            "tfidf_features": (
                self.tfidf.get_feature_names_out().shape[0] if self.tfidf else 0
            ),
            "similarity_matrix_shape": (
                self.cosine_sim.shape if self.cosine_sim is not None else None
            ),
        }

    def force_refresh(self) -> bool:
        """
        Paksa refresh model (bypass cache)

        Returns:
            bool: True jika refresh berhasil
        """
        return self.initialize_model(force_refresh=True)

    def search_innovations(
        self, query: str, top_n: int = 10
    ) -> List[RecommendationResponse]:
        """
        Mencari inovasi berdasarkan query text

        Args:
            query (str): Query pencarian
            top_n (int): Jumlah hasil yang diinginkan

        Returns:
            List[RecommendationResponse]: List hasil pencarian
        """
        if not self.is_initialized or not query.strip():
            return []

        try:
            # Preprocess query
            processed_query = preprocess_text(query)
            if not processed_query:
                return []

            # Transform query menggunakan TF-IDF
            query_vector = self.tfidf.transform([processed_query])

            # Hitung similarity dengan semua dokumen
            similarities = cosine_similarity(
                query_vector, self.tfidf.transform(self.df["processed_text"])
            ).flatten()

            # Get top results
            top_indices = similarities.argsort()[-top_n:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    row = self.df.iloc[idx]
                    rec = self._process_recommendation_row(row, similarities[idx])
                    if rec:
                        results.append(rec)

            return results

        except Exception as e:
            print(f"Error searching innovations: {e}")
            return []

# Singleton instance
recommendation_engine = RecommendationEngine()
