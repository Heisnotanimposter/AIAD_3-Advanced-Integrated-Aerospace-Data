import sqlite3
import asyncio
from typing import Optional
from datetime import datetime
import json

class DatabaseManager:
    """Simple database manager for SQLite"""
    
    def __init__(self, db_path: str = "satellite_data.db"):
        self.db_path = db_path
        self._connection = None
    
    async def connect(self):
        """Create database connection"""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            await self.create_tables()
    
    async def create_tables(self):
        """Create necessary tables"""
        cursor = self._connection.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_url TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                time_horizon INTEGER,
                region TEXT,
                prediction_type TEXT,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        # Satellite images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS satellite_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                satellite_name TEXT NOT NULL,
                image_url TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                region TEXT,
                resolution TEXT,
                file_size INTEGER
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                loss REAL,
                epoch INTEGER,
                model_version TEXT
            )
        ''')
        
        # Analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        self._connection.commit()
    
    async def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        await self.connect()
        cursor = self._connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            return cursor.fetchall()
        else:
            self._connection.commit()
            return cursor.lastrowid
    
    async def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None

# Global database instance
db_manager = DatabaseManager()

async def init_db():
    """Initialize database"""
    await db_manager.connect()
    print("✅ Database initialized successfully")

async def get_db():
    """Get database connection"""
    return db_manager
