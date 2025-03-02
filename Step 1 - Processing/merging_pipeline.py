"""
This script contains all the logic to sequentially process, trim and merge the datasets one by one.
The process is not particularly clean or optimized, but it should be a one-time operation, so anything more would be over-engineering.
"""

import json
import random
import pandas as pd

# Selected datasets paths
set1 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\artermiloff_steam-games-dataset\games_may2024_cleaned.csv"  # Great general metadata foundation
set2 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\antonkozyriev_game-recommendations-on-steam\games.csv"  # grab price_final and price_original
set3 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\fronkongames_steam-games-dataset\games.csv"  # Has interesting features like metacritic score, achievement, recommandations, genres, tags (combine across sets on these?)
set4 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\fronkongames_steam-games-dataset\games.json"  # Unsure. Includes game descriptions an such (NLP?) DLCs, genres, and especially "estimated_owners"
set5 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\nikdavis_steam-store-games_steamspy\steam.csv"  # playtime stats and owners
set6 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\srgiomanhes_steam-games-dataset-2025\steam_games.csv"  # Similar to base set, with 2025 data, but we have to ensure ALL its columns can fully merge in
set7 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\sujaykapadnis_games-on-steam\steamdb.json"  # Some VERY interesting "meta" fields from gamefaqs, stsp, hltb, metacritic and igdb IF complete enough
set8 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\vicentearce_steamdata\game_data.csv"  # FOr the vr tags, whch I THINK other sets don't offer
set9 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\vicentearce_steamdata\final\steam_games.csv"  # Coop, online, workshop support, languages, precise rating, playtime, peak player, owners, has dlc, has demos....
set10 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\souyama_steam-dataset\steam_dataset\steamspy\detailed\steam_spy_detailed.json"  # Owners, CCU, detailled tags
set11 = r"F:\OneDrive\MyDocs\Study\TELUQ\Session 8 - Hiver 2025\SCI 1402\datasets\souyama_steam-dataset\steam_dataset\appinfo\store_data\steam_store_data.json"  # F2P, required age, what is price_overview?, whether games have demos...
