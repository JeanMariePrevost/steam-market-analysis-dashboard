import streamlit as st

import utils

##############################
# Load & Prepare Data
##############################

# Page configuration & custom CSS
st.set_page_config(page_title="Feature Reference")
utils.display_streamlit_custom_navigation()
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        max-width: 1000px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page Title & Description
st.title("Feature Reference")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "General",
        "Categories",
        "Genres",
        "Languages",
        "Tags",
    ]
)

with tab1:
    st.header("General Features")
    st.markdown(
        """
        | Feature Name | Value Type | Range | Comment |
        |--------------|------|--------|-------------|
        | average_time_to_beat | float64 | {min: 0.017, max: 4800.0, mean: 7.852} | Time in hours |
        | categories_count | int | {min: 0, max: 23, mean: 4} | Number of categories listed for the title |
        | controller_support | int | {-1: Unknown, 0: False, 1: True} |  |
        | early_access | int | {-1: Unknown, 0: False, 1: True} |  |
        | gamefaqs_difficulty_rating | int | {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8} | -1: unknown<br>1: "Simple"<br>2: "Simple-Easy"<br>3: "Easy"<br>4: "Easy-Just Right"<br>5: "Just Right"<br>6: "Just Right-Tough"<br>7: "Tough"<br>8: "Tough-Unforgiving"<br>9: "Unforgiving" |
        | genres_count | int | {min: 0, max: 19, mean: 3} | Number of genres listed for the title |
        | has_demos | int | {-1: Unknown, 0: False, 1: True} |  |
        | has_drm | int | {-1: Unknown, 0: False, 1: True} |  |
        | languages_supported_count | float64 | {min: 0.0, max: 94.0, mean: 3.3516297104083366} | Menu and text language |
        | languages_with_full_audio_count | float64 | {min: 0.0, max: 94.0, mean: 1.414927905004241} | Full audio language |
        | monetization_model | int | {-1, 0, 1, 2} |  -1: Unknown<br>0: Paid<br>1: F2P<br>2: Free |
        | price_original | float64 | {min: 0.0, max: 999.98, mean: 9.720832424572881} | Launch price in USD |
        | release_day_of_month | float64 | {min: 1.0, max: 31.0, mean: 15.974421422513025} |  |
        | release_month | int | {min: 1, max: 12, mean: 7} |  |
        | release_year | int | {min: 1997, max: 2025, mean: 2020} |  |
        | required_age | float64 | {min: 0.0, max: 97.0, mean: 2.3154368108566583} |  |
        | runs_on_linux | int | {-1: Unknown, 0: False, 1: True} |  |
        | runs_on_mac | int | {-1: Unknown, 0: False, 1: True} |  |
        | runs_on_steam_deck | int | {-1: Unknown, 0: False, 1: True} |  |
        | runs_on_windows | int | {-1: Unknown, 0: False, 1: True} |  |
        | steam_store_movie_count | int | {min: 0, max: 107, mean: 1} |  |
        | steam_store_screenshot_count | int | {min: 0, max: 183, mean: 9} |  |
        | tags_count | int | {min: 0, max: 37, mean: 11} | Number of tags listed for the title |
        | vr_only | int | {-1: Unknown, 0: False, 1: True} |  |
        | vr_supported | int | {-1: Unknown, 0: False, 1: True} |  |
        
        ## Special Cases
        The featureset includes a large number of categories, genres, languages, and tags. These features all use similar format where a suffix is added to the type's prefix, and the full list can be found in the respective tabs.
        
        | Feature Type | Syntax | Value Type | Range |
        |--------|--------|------|--------|
        | Categories | ~category_[...] | int | {0: False, 1: True} |
        | Genres | ~genre_[...] | int | {0: False, 1: True} |
        | Languages | ~lang_[...] | int | {0: False, 1: True} |
        | Audio Languages | ~lang_audio_[...] | int | {0: False, 1: True} |
        | Languages | ~lang_[...] | int | {-1: Unknown, 0: False, 1: True} |
        | Tags | ~tag_[...] | int | {-1: Unknown, 0: False, 1: True} |
        
        """,
        unsafe_allow_html=True,
    )
with tab2:
    st.markdown(
        """
        ### Categories
        | Feature Name | Value Type | Range |
        |--------------|------|--------|
        | ~category_captions available | int | {0: False, 1: True} |
        | ~category_co-op | int | {0: False, 1: True} |
        | ~category_commentary available | int | {0: False, 1: True} |
        | ~category_cross-platform multiplayer | int | {0: False, 1: True} |
        | ~category_family sharing | int | {0: False, 1: True} |
        | ~category_full controller support | int | {0: False, 1: True} |
        | ~category_hdr available | int | {0: False, 1: True} |
        | ~category_in-app purchases | int | {0: False, 1: True} |
        | ~category_includes level editor | int | {0: False, 1: True} |
        | ~category_includes source sdk | int | {0: False, 1: True} |
        | ~category_lan co-op | int | {0: False, 1: True} |
        | ~category_lan pvp | int | {0: False, 1: True} |
        | ~category_mmo | int | {0: False, 1: True} |
        | ~category_mods (require hl2) | int | {0: False, 1: True} |
        | ~category_mods | int | {0: False, 1: True} |
        | ~category_multi-player | int | {0: False, 1: True} |
        | ~category_online co-op | int | {0: False, 1: True} |
        | ~category_online pvp | int | {0: False, 1: True} |
        | ~category_partial controller support | int | {0: False, 1: True} |
        | ~category_pvp | int | {0: False, 1: True} |
        | ~category_remote play on phone | int | {0: False, 1: True} |
        | ~category_remote play on tablet | int | {0: False, 1: True} |
        | ~category_remote play on tv | int | {0: False, 1: True} |
        | ~category_remote play together | int | {0: False, 1: True} |
        | ~category_shared/split screen | int | {0: False, 1: True} |
        | ~category_shared/split screen co-op | int | {0: False, 1: True} |
        | ~category_shared/split screen pvp | int | {0: False, 1: True} |
        | ~category_single-player | int | {0: False, 1: True} |
        | ~category_stats | int | {0: False, 1: True} |
        | ~category_steam achievements | int | {0: False, 1: True} |
        | ~category_steam cloud | int | {0: False, 1: True} |
        | ~category_steam leaderboards | int | {0: False, 1: True} |
        | ~category_steam trading cards | int | {0: False, 1: True} |
        | ~category_steam turn notifications | int | {0: False, 1: True} |
        | ~category_steam workshop | int | {0: False, 1: True} |
        | ~category_steamvr collectibles | int | {0: False, 1: True} |
        | ~category_tracked controller support | int | {0: False, 1: True} |
        | ~category_valve anti-cheat enabled | int | {0: False, 1: True} |
        | ~category_vr only | int | {0: False, 1: True} |
        | ~category_vr support | int | {0: False, 1: True} |
        | ~category_vr supported | int | {0: False, 1: True} |

        """,
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown(
        """
        ### Genres
        | Feature Name | Value Type | Range |
        |--------------|------|--------|
        | ~genre_360 video | int | {0: False, 1: True} |
        | ~genre_accounting | int | {0: False, 1: True} |
        | ~genre_action | int | {0: False, 1: True} |
        | ~genre_adventure | int | {0: False, 1: True} |
        | ~genre_animation & modeling | int | {0: False, 1: True} |
        | ~genre_audio production | int | {0: False, 1: True} |
        | ~genre_casual | int | {0: False, 1: True} |
        | ~genre_design & illustration | int | {0: False, 1: True} |
        | ~genre_documentary | int | {0: False, 1: True} |
        | ~genre_early access | int | {0: False, 1: True} |
        | ~genre_education | int | {0: False, 1: True} |
        | ~genre_episodic | int | {0: False, 1: True} |
        | ~genre_free to play | int | {0: False, 1: True} |
        | ~genre_game development | int | {0: False, 1: True} |
        | ~genre_gore | int | {0: False, 1: True} |
        | ~genre_indie | int | {0: False, 1: True} |
        | ~genre_massively multiplayer | int | {0: False, 1: True} |
        | ~genre_movie | int | {0: False, 1: True} |
        | ~genre_nudity | int | {0: False, 1: True} |
        | ~genre_photo editing | int | {0: False, 1: True} |
        | ~genre_racing | int | {0: False, 1: True} |
        | ~genre_rpg | int | {0: False, 1: True} |
        | ~genre_sexual content | int | {0: False, 1: True} |
        | ~genre_short | int | {0: False, 1: True} |
        | ~genre_simulation | int | {0: False, 1: True} |
        | ~genre_software training | int | {0: False, 1: True} |
        | ~genre_sports | int | {0: False, 1: True} |
        | ~genre_strategy | int | {0: False, 1: True} |
        | ~genre_tutorial | int | {0: False, 1: True} |
        | ~genre_utilities | int | {0: False, 1: True} |
        | ~genre_video production | int | {0: False, 1: True} |
        | ~genre_violent | int | {0: False, 1: True} |
        | ~genre_web publishing | int | {0: False, 1: True} |

        """,
        unsafe_allow_html=True,
    )
with tab4:
    st.markdown(
        """
        ### Languages
        | Feature Name | Value Type | Range |
        |--------------|------|--------|
        | ~lang_afrikaans | int | {0: False, 1: True} |
        | ~lang_albanian | int | {0: False, 1: True} |
        | ~lang_amharic | int | {0: False, 1: True} |
        | ~lang_arabic | int | {0: False, 1: True} |
        | ~lang_armenian | int | {0: False, 1: True} |
        | ~lang_assamese | int | {0: False, 1: True} |
        | ~lang_audio_afrikaans | int | {0: False, 1: True} |
        | ~lang_audio_albanian | int | {0: False, 1: True} |
        | ~lang_audio_amharic | int | {0: False, 1: True} |
        | ~lang_audio_arabic | int | {0: False, 1: True} |
        | ~lang_audio_armenian | int | {0: False, 1: True} |
        | ~lang_audio_assamese | int | {0: False, 1: True} |
        | ~lang_audio_azerbaijani | int | {0: False, 1: True} |
        | ~lang_audio_bangla | int | {0: False, 1: True} |
        | ~lang_audio_basque | int | {0: False, 1: True} |
        | ~lang_audio_belarusian | int | {0: False, 1: True} |
        | ~lang_audio_bosnian | int | {0: False, 1: True} |
        | ~lang_audio_bulgarian | int | {0: False, 1: True} |
        | ~lang_audio_catalan | int | {0: False, 1: True} |
        | ~lang_audio_cherokee | int | {0: False, 1: True} |
        | ~lang_audio_croatian | int | {0: False, 1: True} |
        | ~lang_audio_czech | int | {0: False, 1: True} |
        | ~lang_audio_danish | int | {0: False, 1: True} |
        | ~lang_audio_dari | int | {0: False, 1: True} |
        | ~lang_audio_dutch | int | {0: False, 1: True} |
        | ~lang_audio_english | int | {0: False, 1: True} |
        | ~lang_audio_estonian | int | {0: False, 1: True} |
        | ~lang_audio_filipino | int | {0: False, 1: True} |
        | ~lang_audio_finnish | int | {0: False, 1: True} |
        | ~lang_audio_french | int | {0: False, 1: True} |
        | ~lang_audio_galician | int | {0: False, 1: True} |
        | ~lang_audio_georgian | int | {0: False, 1: True} |
        | ~lang_audio_german | int | {0: False, 1: True} |
        | ~lang_audio_greek | int | {0: False, 1: True} |
        | ~lang_audio_gujarati | int | {0: False, 1: True} |
        | ~lang_audio_hausa | int | {0: False, 1: True} |
        | ~lang_audio_hebrew | int | {0: False, 1: True} |
        | ~lang_audio_hindi | int | {0: False, 1: True} |
        | ~lang_audio_hungarian | int | {0: False, 1: True} |
        | ~lang_audio_icelandic | int | {0: False, 1: True} |
        | ~lang_audio_igbo | int | {0: False, 1: True} |
        | ~lang_audio_indonesian | int | {0: False, 1: True} |
        | ~lang_audio_irish | int | {0: False, 1: True} |
        | ~lang_audio_italian | int | {0: False, 1: True} |
        | ~lang_audio_japanese | int | {0: False, 1: True} |
        | ~lang_audio_kannada | int | {0: False, 1: True} |
        | ~lang_audio_kazakh | int | {0: False, 1: True} |
        | ~lang_audio_khmer | int | {0: False, 1: True} |
        | ~lang_audio_kinyarwanda | int | {0: False, 1: True} |
        | ~lang_audio_konkani | int | {0: False, 1: True} |
        | ~lang_audio_korean | int | {0: False, 1: True} |
        | ~lang_audio_kyrgyz | int | {0: False, 1: True} |
        | ~lang_audio_latvian | int | {0: False, 1: True} |
        | ~lang_audio_lithuanian | int | {0: False, 1: True} |
        | ~lang_audio_luxembourgish | int | {0: False, 1: True} |
        | ~lang_audio_macedonian | int | {0: False, 1: True} |
        | ~lang_audio_malay | int | {0: False, 1: True} |
        | ~lang_audio_malayalam | int | {0: False, 1: True} |
        | ~lang_audio_maltese | int | {0: False, 1: True} |
        | ~lang_audio_maori | int | {0: False, 1: True} |
        | ~lang_audio_marathi | int | {0: False, 1: True} |
        | ~lang_audio_mongolian | int | {0: False, 1: True} |
        | ~lang_audio_nepali | int | {0: False, 1: True} |
        | ~lang_audio_norwegian | int | {0: False, 1: True} |
        | ~lang_audio_odia | int | {0: False, 1: True} |
        | ~lang_audio_persian | int | {0: False, 1: True} |
        | ~lang_audio_polish | int | {0: False, 1: True} |
        | ~lang_audio_quechua | int | {0: False, 1: True} |
        | ~lang_audio_romanian | int | {0: False, 1: True} |
        | ~lang_audio_russian | int | {0: False, 1: True} |
        | ~lang_audio_scots | int | {0: False, 1: True} |
        | ~lang_audio_serbian | int | {0: False, 1: True} |
        | ~lang_audio_sindhi | int | {0: False, 1: True} |
        | ~lang_audio_sinhala | int | {0: False, 1: True} |
        | ~lang_audio_slovak | int | {0: False, 1: True} |
        | ~lang_audio_slovenian | int | {0: False, 1: True} |
        | ~lang_audio_sorani | int | {0: False, 1: True} |
        | ~lang_audio_sotho | int | {0: False, 1: True} |
        | ~lang_audio_swahili | int | {0: False, 1: True} |
        | ~lang_audio_swedish | int | {0: False, 1: True} |
        | ~lang_audio_tajik | int | {0: False, 1: True} |
        | ~lang_audio_tamil | int | {0: False, 1: True} |
        | ~lang_audio_tatar | int | {0: False, 1: True} |
        | ~lang_audio_telugu | int | {0: False, 1: True} |
        | ~lang_audio_thai | int | {0: False, 1: True} |
        | ~lang_audio_tigrinya | int | {0: False, 1: True} |
        | ~lang_audio_tswana | int | {0: False, 1: True} |
        | ~lang_audio_turkish | int | {0: False, 1: True} |
        | ~lang_audio_turkmen | int | {0: False, 1: True} |
        | ~lang_audio_ukrainian | int | {0: False, 1: True} |
        | ~lang_audio_urdu | int | {0: False, 1: True} |
        | ~lang_audio_uyghur | int | {0: False, 1: True} |
        | ~lang_audio_uzbek | int | {0: False, 1: True} |
        | ~lang_audio_valencian | int | {0: False, 1: True} |
        | ~lang_audio_vietnamese | int | {0: False, 1: True} |
        | ~lang_audio_welsh | int | {0: False, 1: True} |
        | ~lang_audio_wolof | int | {0: False, 1: True} |
        | ~lang_audio_xhosa | int | {0: False, 1: True} |
        | ~lang_audio_yoruba | int | {0: False, 1: True} |
        | ~lang_audio_zulu | int | {0: False, 1: True} |
        | ~lang_azerbaijani | int | {0: False, 1: True} |
        | ~lang_bangla | int | {0: False, 1: True} |
        | ~lang_basque | int | {0: False, 1: True} |
        | ~lang_belarusian | int | {0: False, 1: True} |
        | ~lang_bosnian | int | {0: False, 1: True} |
        | ~lang_bulgarian | int | {0: False, 1: True} |
        | ~lang_catalan | int | {0: False, 1: True} |
        | ~lang_cherokee | int | {0: False, 1: True} |
        | ~lang_croatian | int | {0: False, 1: True} |
        | ~lang_czech | int | {0: False, 1: True} |
        | ~lang_danish | int | {0: False, 1: True} |
        | ~lang_dari | int | {0: False, 1: True} |
        | ~lang_dutch | int | {0: False, 1: True} |
        | ~lang_english | int | {0: False, 1: True} |
        | ~lang_estonian | int | {0: False, 1: True} |
        | ~lang_filipino | int | {0: False, 1: True} |
        | ~lang_finnish | int | {0: False, 1: True} |
        | ~lang_french | int | {0: False, 1: True} |
        | ~lang_galician | int | {0: False, 1: True} |
        | ~lang_georgian | int | {0: False, 1: True} |
        | ~lang_german | int | {0: False, 1: True} |
        | ~lang_greek | int | {0: False, 1: True} |
        | ~lang_gujarati | int | {0: False, 1: True} |
        | ~lang_hausa | int | {0: False, 1: True} |
        | ~lang_hebrew | int | {0: False, 1: True} |
        | ~lang_hindi | int | {0: False, 1: True} |
        | ~lang_hungarian | int | {0: False, 1: True} |
        | ~lang_icelandic | int | {0: False, 1: True} |
        | ~lang_igbo | int | {0: False, 1: True} |
        | ~lang_indonesian | int | {0: False, 1: True} |
        | ~lang_irish | int | {0: False, 1: True} |
        | ~lang_italian | int | {0: False, 1: True} |
        | ~lang_japanese | int | {0: False, 1: True} |
        | ~lang_kannada | int | {0: False, 1: True} |
        | ~lang_kazakh | int | {0: False, 1: True} |
        | ~lang_khmer | int | {0: False, 1: True} |
        | ~lang_kinyarwanda | int | {0: False, 1: True} |
        | ~lang_konkani | int | {0: False, 1: True} |
        | ~lang_korean | int | {0: False, 1: True} |
        | ~lang_kyrgyz | int | {0: False, 1: True} |
        | ~lang_latvian | int | {0: False, 1: True} |
        | ~lang_lithuanian | int | {0: False, 1: True} |
        | ~lang_luxembourgish | int | {0: False, 1: True} |
        | ~lang_macedonian | int | {0: False, 1: True} |
        | ~lang_malay | int | {0: False, 1: True} |
        | ~lang_malayalam | int | {0: False, 1: True} |
        | ~lang_maltese | int | {0: False, 1: True} |
        | ~lang_maori | int | {0: False, 1: True} |
        | ~lang_marathi | int | {0: False, 1: True} |
        | ~lang_mongolian | int | {0: False, 1: True} |
        | ~lang_nepali | int | {0: False, 1: True} |
        | ~lang_norwegian | int | {0: False, 1: True} |
        | ~lang_odia | int | {0: False, 1: True} |
        | ~lang_persian | int | {0: False, 1: True} |
        | ~lang_polish | int | {0: False, 1: True} |
        | ~lang_quechua | int | {0: False, 1: True} |
        | ~lang_romanian | int | {0: False, 1: True} |
        | ~lang_russian | int | {0: False, 1: True} |
        | ~lang_scots | int | {0: False, 1: True} |
        | ~lang_serbian | int | {0: False, 1: True} |
        | ~lang_sindhi | int | {0: False, 1: True} |
        | ~lang_sinhala | int | {0: False, 1: True} |
        | ~lang_slovak | int | {0: False, 1: True} |
        | ~lang_slovenian | int | {0: False, 1: True} |
        | ~lang_sorani | int | {0: False, 1: True} |
        | ~lang_sotho | int | {0: False, 1: True} |
        | ~lang_swahili | int | {0: False, 1: True} |
        | ~lang_swedish | int | {0: False, 1: True} |
        | ~lang_tajik | int | {0: False, 1: True} |
        | ~lang_tamil | int | {0: False, 1: True} |
        | ~lang_tatar | int | {0: False, 1: True} |
        | ~lang_telugu | int | {0: False, 1: True} |
        | ~lang_thai | int | {0: False, 1: True} |
        | ~lang_tigrinya | int | {0: False, 1: True} |
        | ~lang_tswana | int | {0: False, 1: True} |
        | ~lang_turkish | int | {0: False, 1: True} |
        | ~lang_turkmen | int | {0: False, 1: True} |
        | ~lang_ukrainian | int | {0: False, 1: True} |
        | ~lang_urdu | int | {0: False, 1: True} |
        | ~lang_uyghur | int | {0: False, 1: True} |
        | ~lang_uzbek | int | {0: False, 1: True} |
        | ~lang_valencian | int | {0: False, 1: True} |
        | ~lang_vietnamese | int | {0: False, 1: True} |
        | ~lang_welsh | int | {0: False, 1: True} |
        | ~lang_wolof | int | {0: False, 1: True} |
        | ~lang_xhosa | int | {0: False, 1: True} |
        | ~lang_yoruba | int | {0: False, 1: True} |
        | ~lang_zulu | int | {0: False, 1: True} |
    
        """,
        unsafe_allow_html=True,
    )
with tab5:
    st.markdown(
        """
        ### Tags
        | Feature Name | Value Type | Range |
        |--------------|------|--------|
        | ~tag_1980s | int | {0: False, 1: True} |
        | ~tag_1990's | int | {0: False, 1: True} |
        | ~tag_2.5d | int | {0: False, 1: True} |
        | ~tag_2d | int | {0: False, 1: True} |
        | ~tag_2d fighter | int | {0: False, 1: True} |
        | ~tag_2d platformer | int | {0: False, 1: True} |
        | ~tag_360 video | int | {0: False, 1: True} |
        | ~tag_3d | int | {0: False, 1: True} |
        | ~tag_3d fighter | int | {0: False, 1: True} |
        | ~tag_3d platformer | int | {0: False, 1: True} |
        | ~tag_3d vision | int | {0: False, 1: True} |
        | ~tag_4 player local | int | {0: False, 1: True} |
        | ~tag_4x | int | {0: False, 1: True} |
        | ~tag_6dof | int | {0: False, 1: True} |
        | ~tag_8-bit music | int | {0: False, 1: True} |
        | ~tag_abstract | int | {0: False, 1: True} |
        | ~tag_action | int | {0: False, 1: True} |
        | ~tag_action roguelike | int | {0: False, 1: True} |
        | ~tag_action rpg | int | {0: False, 1: True} |
        | ~tag_action rts | int | {0: False, 1: True} |
        | ~tag_action-adventure | int | {0: False, 1: True} |
        | ~tag_addictive | int | {0: False, 1: True} |
        | ~tag_adventure | int | {0: False, 1: True} |
        | ~tag_agriculture | int | {0: False, 1: True} |
        | ~tag_aliens | int | {0: False, 1: True} |
        | ~tag_alternate history | int | {0: False, 1: True} |
        | ~tag_ambient | int | {0: False, 1: True} |
        | ~tag_america | int | {0: False, 1: True} |
        | ~tag_animation & modeling | int | {0: False, 1: True} |
        | ~tag_anime | int | {0: False, 1: True} |
        | ~tag_arcade | int | {0: False, 1: True} |
        | ~tag_archery | int | {0: False, 1: True} |
        | ~tag_arena shooter | int | {0: False, 1: True} |
        | ~tag_artificial intelligence | int | {0: False, 1: True} |
        | ~tag_assassin | int | {0: False, 1: True} |
        | ~tag_asymmetric vr | int | {0: False, 1: True} |
        | ~tag_asynchronous multiplayer | int | {0: False, 1: True} |
        | ~tag_atmospheric | int | {0: False, 1: True} |
        | ~tag_atv | int | {0: False, 1: True} |
        | ~tag_audio production | int | {0: False, 1: True} |
        | ~tag_auto battler | int | {0: False, 1: True} |
        | ~tag_automation | int | {0: False, 1: True} |
        | ~tag_automobile sim | int | {0: False, 1: True} |
        | ~tag_base-building | int | {0: False, 1: True} |
        | ~tag_baseball | int | {0: False, 1: True} |
        | ~tag_based on a novel | int | {0: False, 1: True} |
        | ~tag_basketball | int | {0: False, 1: True} |
        | ~tag_batman | int | {0: False, 1: True} |
        | ~tag_battle royale | int | {0: False, 1: True} |
        | ~tag_beat 'em up | int | {0: False, 1: True} |
        | ~tag_beautiful | int | {0: False, 1: True} |
        | ~tag_benchmark | int | {0: False, 1: True} |
        | ~tag_bikes | int | {0: False, 1: True} |
        | ~tag_birds | int | {0: False, 1: True} |
        | ~tag_blood | int | {0: False, 1: True} |
        | ~tag_bmx | int | {0: False, 1: True} |
        | ~tag_board game | int | {0: False, 1: True} |
        | ~tag_boomer shooter | int | {0: False, 1: True} |
        | ~tag_boss rush | int | {0: False, 1: True} |
        | ~tag_bowling | int | {0: False, 1: True} |
        | ~tag_boxing | int | {0: False, 1: True} |
        | ~tag_building | int | {0: False, 1: True} |
        | ~tag_bullet hell | int | {0: False, 1: True} |
        | ~tag_bullet time | int | {0: False, 1: True} |
        | ~tag_capitalism | int | {0: False, 1: True} |
        | ~tag_card battler | int | {0: False, 1: True} |
        | ~tag_card game | int | {0: False, 1: True} |
        | ~tag_cartoon | int | {0: False, 1: True} |
        | ~tag_cartoony | int | {0: False, 1: True} |
        | ~tag_casual | int | {0: False, 1: True} |
        | ~tag_cats | int | {0: False, 1: True} |
        | ~tag_character action game | int | {0: False, 1: True} |
        | ~tag_character customization | int | {0: False, 1: True} |
        | ~tag_chess | int | {0: False, 1: True} |
        | ~tag_choices matter | int | {0: False, 1: True} |
        | ~tag_choose your own adventure | int | {0: False, 1: True} |
        | ~tag_cinematic | int | {0: False, 1: True} |
        | ~tag_city builder | int | {0: False, 1: True} |
        | ~tag_class-based | int | {0: False, 1: True} |
        | ~tag_classic | int | {0: False, 1: True} |
        | ~tag_clicker | int | {0: False, 1: True} |
        | ~tag_co-op | int | {0: False, 1: True} |
        | ~tag_co-op campaign | int | {0: False, 1: True} |
        | ~tag_coding | int | {0: False, 1: True} |
        | ~tag_cold war | int | {0: False, 1: True} |
        | ~tag_collectathon | int | {0: False, 1: True} |
        | ~tag_colony sim | int | {0: False, 1: True} |
        | ~tag_colorful | int | {0: False, 1: True} |
        | ~tag_combat | int | {0: False, 1: True} |
        | ~tag_combat racing | int | {0: False, 1: True} |
        | ~tag_comedy | int | {0: False, 1: True} |
        | ~tag_comic book | int | {0: False, 1: True} |
        | ~tag_competitive | int | {0: False, 1: True} |
        | ~tag_conspiracy | int | {0: False, 1: True} |
        | ~tag_controller | int | {0: False, 1: True} |
        | ~tag_conversation | int | {0: False, 1: True} |
        | ~tag_cooking | int | {0: False, 1: True} |
        | ~tag_cozy | int | {0: False, 1: True} |
        | ~tag_crafting | int | {0: False, 1: True} |
        | ~tag_creature collector | int | {0: False, 1: True} |
        | ~tag_cricket | int | {0: False, 1: True} |
        | ~tag_crime | int | {0: False, 1: True} |
        | ~tag_crowdfunded | int | {0: False, 1: True} |
        | ~tag_crpg | int | {0: False, 1: True} |
        | ~tag_cute | int | {0: False, 1: True} |
        | ~tag_cyberpunk | int | {0: False, 1: True} |
        | ~tag_cycling | int | {0: False, 1: True} |
        | ~tag_dark | int | {0: False, 1: True} |
        | ~tag_dark comedy | int | {0: False, 1: True} |
        | ~tag_dark fantasy | int | {0: False, 1: True} |
        | ~tag_dark humor | int | {0: False, 1: True} |
        | ~tag_dating sim | int | {0: False, 1: True} |
        | ~tag_deckbuilding | int | {0: False, 1: True} |
        | ~tag_demons | int | {0: False, 1: True} |
        | ~tag_design & illustration | int | {0: False, 1: True} |
        | ~tag_destruction | int | {0: False, 1: True} |
        | ~tag_detective | int | {0: False, 1: True} |
        | ~tag_difficult | int | {0: False, 1: True} |
        | ~tag_dinosaurs | int | {0: False, 1: True} |
        | ~tag_diplomacy | int | {0: False, 1: True} |
        | ~tag_documentary | int | {0: False, 1: True} |
        | ~tag_dog | int | {0: False, 1: True} |
        | ~tag_dragons | int | {0: False, 1: True} |
        | ~tag_drama | int | {0: False, 1: True} |
        | ~tag_driving | int | {0: False, 1: True} |
        | ~tag_dungeon crawler | int | {0: False, 1: True} |
        | ~tag_dungeons & dragons | int | {0: False, 1: True} |
        | ~tag_dwarf | int | {0: False, 1: True} |
        | ~tag_dynamic narration | int | {0: False, 1: True} |
        | ~tag_dystopian | int | {0: False, 1: True} |
        | ~tag_e-sports | int | {0: False, 1: True} |
        | ~tag_early access | int | {0: False, 1: True} |
        | ~tag_economy | int | {0: False, 1: True} |
        | ~tag_education | int | {0: False, 1: True} |
        | ~tag_electronic | int | {0: False, 1: True} |
        | ~tag_electronic music | int | {0: False, 1: True} |
        | ~tag_elf | int | {0: False, 1: True} |
        | ~tag_emotional | int | {0: False, 1: True} |
        | ~tag_epic | int | {0: False, 1: True} |
        | ~tag_episodic | int | {0: False, 1: True} |
        | ~tag_escape room | int | {0: False, 1: True} |
        | ~tag_esports | int | {0: False, 1: True} |
        | ~tag_experience | int | {0: False, 1: True} |
        | ~tag_experimental | int | {0: False, 1: True} |
        | ~tag_exploration | int | {0: False, 1: True} |
        | ~tag_extraction shooter | int | {0: False, 1: True} |
        | ~tag_faith | int | {0: False, 1: True} |
        | ~tag_family friendly | int | {0: False, 1: True} |
        | ~tag_fantasy | int | {0: False, 1: True} |
        | ~tag_farming | int | {0: False, 1: True} |
        | ~tag_farming sim | int | {0: False, 1: True} |
        | ~tag_fast-paced | int | {0: False, 1: True} |
        | ~tag_feature film | int | {0: False, 1: True} |
        | ~tag_female protagonist | int | {0: False, 1: True} |
        | ~tag_fighting | int | {0: False, 1: True} |
        | ~tag_first-person | int | {0: False, 1: True} |
        | ~tag_fishing | int | {0: False, 1: True} |
        | ~tag_flight | int | {0: False, 1: True} |
        | ~tag_fmv | int | {0: False, 1: True} |
        | ~tag_football (american) | int | {0: False, 1: True} |
        | ~tag_football (soccer) | int | {0: False, 1: True} |
        | ~tag_football | int | {0: False, 1: True} |
        | ~tag_foreign | int | {0: False, 1: True} |
        | ~tag_fox | int | {0: False, 1: True} |
        | ~tag_fps | int | {0: False, 1: True} |
        | ~tag_free to play | int | {0: False, 1: True} |
        | ~tag_funny | int | {0: False, 1: True} |
        | ~tag_futuristic | int | {0: False, 1: True} |
        | ~tag_gambling | int | {0: False, 1: True} |
        | ~tag_game development | int | {0: False, 1: True} |
        | ~tag_gamemaker | int | {0: False, 1: True} |
        | ~tag_games workshop | int | {0: False, 1: True} |
        | ~tag_gaming | int | {0: False, 1: True} |
        | ~tag_god game | int | {0: False, 1: True} |
        | ~tag_golf | int | {0: False, 1: True} |
        | ~tag_gore | int | {0: False, 1: True} |
        | ~tag_gothic | int | {0: False, 1: True} |
        | ~tag_grand strategy | int | {0: False, 1: True} |
        | ~tag_great soundtrack | int | {0: False, 1: True} |
        | ~tag_grid-based movement | int | {0: False, 1: True} |
        | ~tag_gun customization | int | {0: False, 1: True} |
        | ~tag_hack and slash | int | {0: False, 1: True} |
        | ~tag_hacking | int | {0: False, 1: True} |
        | ~tag_hand-drawn | int | {0: False, 1: True} |
        | ~tag_hardware | int | {0: False, 1: True} |
        | ~tag_heist | int | {0: False, 1: True} |
        | ~tag_hentai | int | {0: False, 1: True} |
        | ~tag_hero shooter | int | {0: False, 1: True} |
        | ~tag_hex grid | int | {0: False, 1: True} |
        | ~tag_hidden object | int | {0: False, 1: True} |
        | ~tag_historical | int | {0: False, 1: True} |
        | ~tag_hobby sim | int | {0: False, 1: True} |
        | ~tag_hockey | int | {0: False, 1: True} |
        | ~tag_horror | int | {0: False, 1: True} |
        | ~tag_horses | int | {0: False, 1: True} |
        | ~tag_hunting | int | {0: False, 1: True} |
        | ~tag_idler | int | {0: False, 1: True} |
        | ~tag_illuminati | int | {0: False, 1: True} |
        | ~tag_immersive | int | {0: False, 1: True} |
        | ~tag_immersive sim | int | {0: False, 1: True} |
        | ~tag_indie | int | {0: False, 1: True} |
        | ~tag_instrumental music | int | {0: False, 1: True} |
        | ~tag_intentionally awkward controls | int | {0: False, 1: True} |
        | ~tag_interactive fiction | int | {0: False, 1: True} |
        | ~tag_inventory management | int | {0: False, 1: True} |
        | ~tag_investigation | int | {0: False, 1: True} |
        | ~tag_isometric | int | {0: False, 1: True} |
        | ~tag_jet | int | {0: False, 1: True} |
        | ~tag_job simulator | int | {0: False, 1: True} |
        | ~tag_jrpg | int | {0: False, 1: True} |
        | ~tag_jump scare | int | {0: False, 1: True} |
        | ~tag_kickstarter | int | {0: False, 1: True} |
        | ~tag_lara croft | int | {0: False, 1: True} |
        | ~tag_lego | int | {0: False, 1: True} |
        | ~tag_lemmings | int | {0: False, 1: True} |
        | ~tag_level editor | int | {0: False, 1: True} |
        | ~tag_lgbtq+ | int | {0: False, 1: True} |
        | ~tag_life sim | int | {0: False, 1: True} |
        | ~tag_linear | int | {0: False, 1: True} |
        | ~tag_local co-op | int | {0: False, 1: True} |
        | ~tag_local multiplayer | int | {0: False, 1: True} |
        | ~tag_logic | int | {0: False, 1: True} |
        | ~tag_loot | int | {0: False, 1: True} |
        | ~tag_looter shooter | int | {0: False, 1: True} |
        | ~tag_lore-rich | int | {0: False, 1: True} |
        | ~tag_lovecraftian | int | {0: False, 1: True} |
        | ~tag_magic | int | {0: False, 1: True} |
        | ~tag_mahjong | int | {0: False, 1: True} |
        | ~tag_management | int | {0: False, 1: True} |
        | ~tag_mars | int | {0: False, 1: True} |
        | ~tag_martial arts | int | {0: False, 1: True} |
        | ~tag_massively multiplayer | int | {0: False, 1: True} |
        | ~tag_match 3 | int | {0: False, 1: True} |
        | ~tag_mature | int | {0: False, 1: True} |
        | ~tag_mechs | int | {0: False, 1: True} |
        | ~tag_medical sim | int | {0: False, 1: True} |
        | ~tag_medieval | int | {0: False, 1: True} |
        | ~tag_memes | int | {0: False, 1: True} |
        | ~tag_metroidvania | int | {0: False, 1: True} |
        | ~tag_military | int | {0: False, 1: True} |
        | ~tag_mini golf | int | {0: False, 1: True} |
        | ~tag_minigames | int | {0: False, 1: True} |
        | ~tag_minimalist | int | {0: False, 1: True} |
        | ~tag_mining | int | {0: False, 1: True} |
        | ~tag_mmorpg | int | {0: False, 1: True} |
        | ~tag_moba | int | {0: False, 1: True} |
        | ~tag_mod | int | {0: False, 1: True} |
        | ~tag_moddable | int | {0: False, 1: True} |
        | ~tag_modern | int | {0: False, 1: True} |
        | ~tag_motocross | int | {0: False, 1: True} |
        | ~tag_motorbike | int | {0: False, 1: True} |
        | ~tag_mouse only | int | {0: False, 1: True} |
        | ~tag_movie | int | {0: False, 1: True} |
        | ~tag_multiplayer | int | {0: False, 1: True} |
        | ~tag_multiple endings | int | {0: False, 1: True} |
        | ~tag_music | int | {0: False, 1: True} |
        | ~tag_music-based procedural generation | int | {0: False, 1: True} |
        | ~tag_musou | int | {0: False, 1: True} |
        | ~tag_mystery | int | {0: False, 1: True} |
        | ~tag_mystery dungeon | int | {0: False, 1: True} |
        | ~tag_mythology | int | {0: False, 1: True} |
        | ~tag_narration | int | {0: False, 1: True} |
        | ~tag_narrative | int | {0: False, 1: True} |
        | ~tag_nature | int | {0: False, 1: True} |
        | ~tag_naval | int | {0: False, 1: True} |
        | ~tag_naval combat | int | {0: False, 1: True} |
        | ~tag_ninja | int | {0: False, 1: True} |
        | ~tag_noir | int | {0: False, 1: True} |
        | ~tag_nonlinear | int | {0: False, 1: True} |
        | ~tag_nostalgia | int | {0: False, 1: True} |
        | ~tag_nsfw | int | {0: False, 1: True} |
        | ~tag_nudity | int | {0: False, 1: True} |
        | ~tag_offroad | int | {0: False, 1: True} |
        | ~tag_old school | int | {0: False, 1: True} |
        | ~tag_on-rails shooter | int | {0: False, 1: True} |
        | ~tag_online co-op | int | {0: False, 1: True} |
        | ~tag_open world | int | {0: False, 1: True} |
        | ~tag_open world survival craft | int | {0: False, 1: True} |
        | ~tag_otome | int | {0: False, 1: True} |
        | ~tag_outbreak sim | int | {0: False, 1: True} |
        | ~tag_parkour | int | {0: False, 1: True} |
        | ~tag_parody | int | {0: False, 1: True} |
        | ~tag_party | int | {0: False, 1: True} |
        | ~tag_party game | int | {0: False, 1: True} |
        | ~tag_party-based rpg | int | {0: False, 1: True} |
        | ~tag_perma death | int | {0: False, 1: True} |
        | ~tag_philisophical | int | {0: False, 1: True} |
        | ~tag_philosophical | int | {0: False, 1: True} |
        | ~tag_photo editing | int | {0: False, 1: True} |
        | ~tag_physics | int | {0: False, 1: True} |
        | ~tag_pinball | int | {0: False, 1: True} |
        | ~tag_pirates | int | {0: False, 1: True} |
        | ~tag_pixel graphics | int | {0: False, 1: True} |
        | ~tag_platformer | int | {0: False, 1: True} |
        | ~tag_point & click | int | {0: False, 1: True} |
        | ~tag_political | int | {0: False, 1: True} |
        | ~tag_political sim | int | {0: False, 1: True} |
        | ~tag_politics | int | {0: False, 1: True} |
        | ~tag_pool | int | {0: False, 1: True} |
        | ~tag_post-apocalyptic | int | {0: False, 1: True} |
        | ~tag_precision platformer | int | {0: False, 1: True} |
        | ~tag_procedural generation | int | {0: False, 1: True} |
        | ~tag_programming | int | {0: False, 1: True} |
        | ~tag_psychedelic | int | {0: False, 1: True} |
        | ~tag_psychological | int | {0: False, 1: True} |
        | ~tag_psychological horror | int | {0: False, 1: True} |
        | ~tag_puzzle | int | {0: False, 1: True} |
        | ~tag_puzzle platformer | int | {0: False, 1: True} |
        | ~tag_pve | int | {0: False, 1: True} |
        | ~tag_pvp | int | {0: False, 1: True} |
        | ~tag_quick-time events | int | {0: False, 1: True} |
        | ~tag_racing | int | {0: False, 1: True} |
        | ~tag_real time tactics | int | {0: False, 1: True} |
        | ~tag_real-time | int | {0: False, 1: True} |
        | ~tag_real-time with pause | int | {0: False, 1: True} |
        | ~tag_realistic | int | {0: False, 1: True} |
        | ~tag_reboot | int | {0: False, 1: True} |
        | ~tag_relaxing | int | {0: False, 1: True} |
        | ~tag_remake | int | {0: False, 1: True} |
        | ~tag_replay value | int | {0: False, 1: True} |
        | ~tag_resource management | int | {0: False, 1: True} |
        | ~tag_retro | int | {0: False, 1: True} |
        | ~tag_rhythm | int | {0: False, 1: True} |
        | ~tag_robots | int | {0: False, 1: True} |
        | ~tag_rock music | int | {0: False, 1: True} |
        | ~tag_roguelike | int | {0: False, 1: True} |
        | ~tag_roguelike deckbuilder | int | {0: False, 1: True} |
        | ~tag_roguelite | int | {0: False, 1: True} |
        | ~tag_roguevania | int | {0: False, 1: True} |
        | ~tag_romance | int | {0: False, 1: True} |
        | ~tag_rome | int | {0: False, 1: True} |
        | ~tag_rpg | int | {0: False, 1: True} |
        | ~tag_rpgmaker | int | {0: False, 1: True} |
        | ~tag_rts | int | {0: False, 1: True} |
        | ~tag_rugby | int | {0: False, 1: True} |
        | ~tag_runner | int | {0: False, 1: True} |
        | ~tag_sailing | int | {0: False, 1: True} |
        | ~tag_sandbox | int | {0: False, 1: True} |
        | ~tag_satire | int | {0: False, 1: True} |
        | ~tag_sci-fi | int | {0: False, 1: True} |
        | ~tag_science | int | {0: False, 1: True} |
        | ~tag_score attack | int | {0: False, 1: True} |
        | ~tag_sequel | int | {0: False, 1: True} |
        | ~tag_sexual content | int | {0: False, 1: True} |
        | ~tag_shoot 'em up | int | {0: False, 1: True} |
        | ~tag_shooter | int | {0: False, 1: True} |
        | ~tag_shop keeper | int | {0: False, 1: True} |
        | ~tag_short | int | {0: False, 1: True} |
        | ~tag_side scroller | int | {0: False, 1: True} |
        | ~tag_silent protagonist | int | {0: False, 1: True} |
        | ~tag_simulation | int | {0: False, 1: True} |
        | ~tag_singleplayer | int | {0: False, 1: True} |
        | ~tag_skateboarding | int | {0: False, 1: True} |
        | ~tag_skating | int | {0: False, 1: True} |
        | ~tag_skiing | int | {0: False, 1: True} |
        | ~tag_sniper | int | {0: False, 1: True} |
        | ~tag_snooker | int | {0: False, 1: True} |
        | ~tag_snow | int | {0: False, 1: True} |
        | ~tag_snowboarding | int | {0: False, 1: True} |
        | ~tag_soccer | int | {0: False, 1: True} |
        | ~tag_social deduction | int | {0: False, 1: True} |
        | ~tag_software | int | {0: False, 1: True} |
        | ~tag_software training | int | {0: False, 1: True} |
        | ~tag_sokoban | int | {0: False, 1: True} |
        | ~tag_solitaire | int | {0: False, 1: True} |
        | ~tag_souls-like | int | {0: False, 1: True} |
        | ~tag_soundtrack | int | {0: False, 1: True} |
        | ~tag_space | int | {0: False, 1: True} |
        | ~tag_space sim | int | {0: False, 1: True} |
        | ~tag_spaceships | int | {0: False, 1: True} |
        | ~tag_spectacle fighter | int | {0: False, 1: True} |
        | ~tag_spelling | int | {0: False, 1: True} |
        | ~tag_split screen | int | {0: False, 1: True} |
        | ~tag_sports | int | {0: False, 1: True} |
        | ~tag_star wars | int | {0: False, 1: True} |
        | ~tag_stealth | int | {0: False, 1: True} |
        | ~tag_steam machine | int | {0: False, 1: True} |
        | ~tag_steampunk | int | {0: False, 1: True} |
        | ~tag_story rich | int | {0: False, 1: True} |
        | ~tag_strategy | int | {0: False, 1: True} |
        | ~tag_strategy rpg | int | {0: False, 1: True} |
        | ~tag_stylized | int | {0: False, 1: True} |
        | ~tag_submarine | int | {0: False, 1: True} |
        | ~tag_superhero | int | {0: False, 1: True} |
        | ~tag_supernatural | int | {0: False, 1: True} |
        | ~tag_surreal | int | {0: False, 1: True} |
        | ~tag_survival | int | {0: False, 1: True} |
        | ~tag_survival horror | int | {0: False, 1: True} |
        | ~tag_swordplay | int | {0: False, 1: True} |
        | ~tag_tabletop | int | {0: False, 1: True} |
        | ~tag_tactical | int | {0: False, 1: True} |
        | ~tag_tactical rpg | int | {0: False, 1: True} |
        | ~tag_tanks | int | {0: False, 1: True} |
        | ~tag_team-based | int | {0: False, 1: True} |
        | ~tag_tennis | int | {0: False, 1: True} |
        | ~tag_text-based | int | {0: False, 1: True} |
        | ~tag_third person | int | {0: False, 1: True} |
        | ~tag_third-person shooter | int | {0: False, 1: True} |
        | ~tag_thriller | int | {0: False, 1: True} |
        | ~tag_tile-matching | int | {0: False, 1: True} |
        | ~tag_time attack | int | {0: False, 1: True} |
        | ~tag_time management | int | {0: False, 1: True} |
        | ~tag_time manipulation | int | {0: False, 1: True} |
        | ~tag_time travel | int | {0: False, 1: True} |
        | ~tag_top-down | int | {0: False, 1: True} |
        | ~tag_top-down shooter | int | {0: False, 1: True} |
        | ~tag_touch-friendly | int | {0: False, 1: True} |
        | ~tag_tower defense | int | {0: False, 1: True} |
        | ~tag_trackir | int | {0: False, 1: True} |
        | ~tag_trading | int | {0: False, 1: True} |
        | ~tag_trading card game | int | {0: False, 1: True} |
        | ~tag_traditional roguelike | int | {0: False, 1: True} |
        | ~tag_trains | int | {0: False, 1: True} |
        | ~tag_transhumanism | int | {0: False, 1: True} |
        | ~tag_transportation | int | {0: False, 1: True} |
        | ~tag_trivia | int | {0: False, 1: True} |
        | ~tag_turn-based | int | {0: False, 1: True} |
        | ~tag_turn-based combat | int | {0: False, 1: True} |
        | ~tag_turn-based strategy | int | {0: False, 1: True} |
        | ~tag_turn-based tactics | int | {0: False, 1: True} |
        | ~tag_tutorial | int | {0: False, 1: True} |
        | ~tag_twin stick shooter | int | {0: False, 1: True} |
        | ~tag_typing | int | {0: False, 1: True} |
        | ~tag_underground | int | {0: False, 1: True} |
        | ~tag_underwater | int | {0: False, 1: True} |
        | ~tag_unforgiving | int | {0: False, 1: True} |
        | ~tag_utilities | int | {0: False, 1: True} |
        | ~tag_vampire | int | {0: False, 1: True} |
        | ~tag_vehicular combat | int | {0: False, 1: True} |
        | ~tag_video production | int | {0: False, 1: True} |
        | ~tag_vikings | int | {0: False, 1: True} |
        | ~tag_villain protagonist | int | {0: False, 1: True} |
        | ~tag_violent | int | {0: False, 1: True} |
        | ~tag_visual novel | int | {0: False, 1: True} |
        | ~tag_voice control | int | {0: False, 1: True} |
        | ~tag_volleyball | int | {0: False, 1: True} |
        | ~tag_voxel | int | {0: False, 1: True} |
        | ~tag_vr | int | {0: False, 1: True} |
        | ~tag_vr only | int | {0: False, 1: True} |
        | ~tag_walking simulator | int | {0: False, 1: True} |
        | ~tag_war | int | {0: False, 1: True} |
        | ~tag_wargame | int | {0: False, 1: True} |
        | ~tag_warhammer 40k | int | {0: False, 1: True} |
        | ~tag_web publishing | int | {0: False, 1: True} |
        | ~tag_well-written | int | {0: False, 1: True} |
        | ~tag_werewolves | int | {0: False, 1: True} |
        | ~tag_western | int | {0: False, 1: True} |
        | ~tag_wholesome | int | {0: False, 1: True} |
        | ~tag_word game | int | {0: False, 1: True} |
        | ~tag_world war i | int | {0: False, 1: True} |
        | ~tag_world war ii | int | {0: False, 1: True} |
        | ~tag_wrestling | int | {0: False, 1: True} |
        | ~tag_zombies | int | {0: False, 1: True} |
        """,
        unsafe_allow_html=True,
    )
