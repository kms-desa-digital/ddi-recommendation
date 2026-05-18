from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_ENV: str = "prod"
    
    # MongoDB Config (Database utama untuk sistem rekomendasi)
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "db_desa"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Menghindari error jika ada variabel ekstra di .env


settings = Settings()


