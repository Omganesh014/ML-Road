"""
TruthBridge Database Initialization Script
Run once to create the SQLite database schema and seed sample locations.
Usage: python init_db.py
"""

import asyncio
import json
import yaml
from pathlib import Path
from database import Database


SAMPLE_LOCATIONS = [
    ("bridge_1", 0.3),
    ("bridge_2", 0.4),
    ("bridge_3", 0.5),
    ("highway_101", 0.6),
    ("highway_102", 0.5),
    ("highway_103", 0.7),
    ("interstate_5", 0.4),
    ("interstate_10", 0.6),
    ("main_street", 0.2),
    ("downtown_bridge", 0.8),
    ("old_bridge_7", 0.9),
    ("construction_zone_a", 0.5),
]


async def init_database():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_path = Path(__file__).parent / config["database"]["path"]
    db = Database(str(db_path))

    print("Initializing database...")
    await db.init_db()
    print(f"Created database at: {db_path}")

    print("Seeding sample locations...")
    for name, base_risk in SAMPLE_LOCATIONS:
        await db.set_location_risk(name, base_risk)
        print(f"  Added location: {name} (base_risk={base_risk})")

    print("\nDatabase initialization complete!")
    print(f"Total locations seeded: {len(SAMPLE_LOCATIONS)}")


if __name__ == "__main__":
    asyncio.run(init_database())