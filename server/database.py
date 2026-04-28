"""
TruthBridge Database Module
Async SQLite operations for damage reports and locations.
"""

import json
import aiosqlite
from pathlib import Path
from typing import Optional
from datetime import datetime


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS damage_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT NOT NULL,
                    damage_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT,
                    coverage_percent REAL NOT NULL,
                    severity TEXT NOT NULL,
                    location TEXT,
                    risk_score REAL,
                    rainfall_mm REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    base_risk REAL DEFAULT 0.5
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_hash ON damage_reports(image_hash)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_location ON damage_reports(location)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_severity ON damage_reports(severity)
            """)
            await db.commit()

    async def insert_report(
        self,
        image_hash: str,
        damage_type: str,
        confidence: float,
        bbox: list,
        coverage_percent: float,
        severity: str,
        location: Optional[str] = None,
        risk_score: Optional[float] = None,
        rainfall_mm: Optional[float] = None,
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO damage_reports
                (image_hash, damage_type, confidence, bbox, coverage_percent,
                 severity, location, risk_score, rainfall_mm, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_hash,
                damage_type,
                confidence,
                json.dumps(bbox),
                coverage_percent,
                severity,
                location,
                risk_score,
                rainfall_mm,
                datetime.utcnow().isoformat(),
            ))
            await db.commit()
            return cursor.lastrowid

    async def get_damage_count(self, location: str, days: int = 30) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT COUNT(*) FROM damage_reports
                WHERE location = ?
                AND created_at >= datetime('now', '-' || ? || ' days')
            """, (location, days))
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_location_risk(self, location: str) -> Optional[float]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT base_risk FROM locations WHERE name = ?",
                (location,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def set_location_risk(self, location: str, base_risk: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO locations (name, base_risk)
                VALUES (?, ?)
            """, (location, base_risk))
            await db.commit()

    async def get_recent_reports(self, location: Optional[str] = None, limit: int = 10):
        async with aiosqlite.connect(self.db_path) as db:
            if location:
                cursor = await db.execute("""
                    SELECT id, image_hash, damage_type, confidence, bbox,
                           coverage_percent, severity, location, risk_score, created_at
                    FROM damage_reports
                    WHERE location = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (location, limit))
            else:
                cursor = await db.execute("""
                    SELECT id, image_hash, damage_type, confidence, bbox,
                           coverage_percent, severity, location, risk_score, created_at
                    FROM damage_reports
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "image_hash": r[1],
                    "damage_type": r[2],
                    "confidence": r[3],
                    "bbox": json.loads(r[4]) if r[4] else [],
                    "coverage_percent": r[5],
                    "severity": r[6],
                    "location": r[7],
                    "risk_score": r[8],
                    "created_at": r[9],
                }
                for r in rows
            ]

    async def get_stats(self):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total_reports,
                    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_count,
                    SUM(CASE WHEN severity = 'moderate' THEN 1 ELSE 0 END) as moderate_count,
                    SUM(CASE WHEN severity = 'minor' THEN 1 ELSE 0 END) as minor_count,
                    COUNT(DISTINCT location) as unique_locations
                FROM damage_reports
            """)
            row = await cursor.fetchone()
            return {
                "total_reports": row[0] or 0,
                "critical_count": row[1] or 0,
                "moderate_count": row[2] or 0,
                "minor_count": row[3] or 0,
                "unique_locations": row[4] or 0,
            }