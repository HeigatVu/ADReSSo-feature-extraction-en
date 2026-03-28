from functools import lru_cache
from typing import Counter
import spacy
import math
from ideadensity import depid
from textblob import TextBlob
from spacy.tokens import Doc
import textstat

# Traditional Linguistic features
@lru_cache
def __load_spacy(lang:str="en") -> None:
    """Support calling one time spacy model -> faster speed
    """
    if lang == "en":
        return spacy.load("en_core_web_sm")
    else:
        print("Current does not support")

def clean_and_tokenize_spacy(transcript:str, lang:str="en") -> tuple[list, str]:
    nlp = __load_spacy(lang)
    doc = nlp(transcript or "")
    words = []
    for word in doc:
        if word.is_alpha:
            words.append(word)

    return words, doc


def lexical_richness(transcript:str, lang:str="en") -> tuple[float]:
    """ Meansure lexical richness
    """
    
    words, _ = clean_and_tokenize_spacy(transcript, lang)
    
    N = len(words) # Total num of word
    freqs = Counter(words) # Frequency of each word in transcript
    V = len(freqs) # Unique words

    # Corrected type-token ratio // https://lexicalrichness.readthedocs.io/en/latest/docstring_docs.html
    if N > 0:
        cttr = round(V/math.sqrt(2*N), 2)
    else:
        cttr = 0

    # Brunet // https://arxiv.org/pdf/2109.11010
    if (N > 0) and (V > 0):
        brunet = round(N**(V**(-0.165)), 2)
    else:
        brunet = 0

    # Honore statistic // https://arxiv.org/pdf/2109.11010 -> performance None when testing -> skip
    # v1 = 0
    # for c in freqs.values():
    #     if c == 1:
    #         v1 += 1
    # if V > v1 and N>0 and V>0:
    #     honore = round((100*math.log(N))/(1 - (v1/V)), 2)
    # else:
    #     honore = float("nan")

    # Standardised Entropy in linguistic /‌/ https://arxiv.org/pdf/2109.11010
    entropy = 0.0
    if N > 1:
        for count in freqs.values():
            p_xi = count / N
            entropy -= p_xi * math.log2(p_xi)
        std_entropy = round((entropy / math.log(N) + 1e-5), 5)
    else:
        std_entropy = 0.0

    # Idea density // https://aclanthology.org/K17-1033.pdf
    density,_,_ = depid(transcript or "", is_depid_r=True)
    pidensity = round(float(density), 5)

    return cttr, brunet, std_entropy, pidensity


## Extract pos_tagged, polarity, subjectivity, pos rate
def polarity(transcript:str) -> float:
    """ Extract polarity in transcript
    """
    return TextBlob(transcript.text).sentiment.polarity

def subjectivity(transcript:str) -> float:
    """ Extract subjectivity in transcript
    """
    return TextBlob(transcript.text).sentiment.subjectivity

Doc.set_extension("polarity", getter=polarity)
Doc.set_extension("subjectivity", getter=subjectivity)

def pos_polarity_subjectivity(transcript:str, lang:str="en") -> tuple[list[tuple], float, float]:
    """ Including for POS tag, polarity and subjectivity
    """

    nlp = __load_spacy(lang)
    doc = nlp(transcript or "")

    polarity = doc._.polarity
    subjectivity = doc._.subjectivity

    pos_tagged_data = []
    for token in doc:
        pos_tagged_data.append((token.text, token.pos_))

    return pos_tagged_data, polarity, subjectivity


def tag_count(pos_tags:list[tuple]) -> dict[str:int,float]:
    """ Counting all word categories on POS tag
    """

    tags = []
    for tag in pos_tags:
        if (len(tag) == 2):
            tags.append(tag[1])

    pos_counts = Counter(tags)

    counts = {
        "verbs": pos_counts.get("VERB", 0),
        "nouns": pos_counts.get("NOUN", 0),
        "pronouns": pos_counts.get("PRON", 0),
        "adjectives": pos_counts.get("ADJ", 0),
        "adverbs": pos_counts.get("ADV", 0),
        "interjections": pos_counts.get("INTJ", 0),
        "determiners": pos_counts.get("DET", 0),
        "conjunctions": pos_counts.get("CCONJ", 0),
        "prepositions": pos_counts.get("ADP", 0),
        "auxiliary_verbs": pos_counts.get("AUX", 0),
        "particles": pos_counts.get("PART", 0),
        "numbers": pos_counts.get("NUM", 0),  
    }

    total_counts = sum(counts.values())
    open_class_words = counts["verbs"] + counts["nouns"] + counts["adjectives"] + counts["adverbs"]
    closed_class_words = total_counts - open_class_words

    if closed_class_words != 0:
        content_density = open_class_words/closed_class_words
    else:
        content_density = float("nan")

## Voice Quality and Phonation
# def get_harmonics_to_noise_ratio_attributes(
#                                             audio_file:parselmouth.Sound,
#                                             harmonic_type:str="preferred",
#                                             time_step:float=0.01,
#                                             min_time:float=0.0,
#                                             max_time:float=0.0,
#                                             minimum_pit

    counts["total_counts"] = total_counts
    counts["open_class_words"] = open_class_words
    counts["closed_class_words"] = closed_class_words
    counts["content_density"] = content_density

    return counts

def evaluate_pos_rate(tag_counts:dict[str:int,float]) -> dict[str:float]:
    """ Evaluate each type of word rate
    """

    rates = dict()
    total_counts = tag_counts.get("total_counts", 0)

    for pos_category, count in tag_counts.items():
        if pos_category not in ["total_counts", "open_class_words", "closed_class_words", "content_density"]:
            if total_counts > 0:
                rates[f"{pos_category}_rate"] = round((count/total_counts), 5)
            else:
                rates[f"{pos_category}_rate"] = 0.0

    return rates

## Evaluate disfluency
def count_disfluency(transcript:str, lang="en") -> int:
    """ Evaluate fuency
    """

    nlp = __load_spacy(lang)
    doc = nlp(transcript or "")

    # Words that are almost universally used as fillers/hesitations
    pure_fillers = {'uh', 'uhh', 'um', 'umm', 'oh', 'ohh', 'hm', 'hmm', 'er', 'erm'}
    
    # Words that have valid grammatical roles, but become disfluencies when used as interjections
    ambiguous_fillers = {'like', 'so', 'well', 'right', 'okay', 'alright', 'actually', 'basically'}

    disfluency_count = 0
    for token in doc:
        word = token.text.lower()
        
        if word in pure_fillers:
            disfluency_count += 1
        elif word in ambiguous_fillers and token.pos_ == "INTJ":
            disfluency_count += 1
        

    
    return disfluency_count

# Readability
def evaluate_readability(transcript:str) -> tuple[float, float, float, float, int]:
    """ Evaluate readability of transcript leverge scores: dale chall, flesch, coleman liau index, automated readability index, r-time and syllables
    """

    dale_chall = textstat.dale_chall_readability_score(transcript)
    flesch = textstat.flesch_reading_ease(transcript)
    coleman_liau_index = textstat.coleman_liau_index(transcript)
    r_time = textstat.reading_time(transcript, ms_per_char=52) # assume that normal WPM=260 and average word has 5 -> ms_per_char=52
    syllables = textstat.syllable_count(transcript)

    return dale_chall, flesch, coleman_liau_index, r_time, syllables

# Evaluate each rate
def evaluate_deixis(transcripti:str, lang:str="en") -> tuple[float]:
    """ Evaluate defiction by person, spatial deixis rate in the transcript
    """

    nlp = __load_spacy(lang)
    doc = nlp(transcripti or "")

    person_deixis_tags = {"PRP", "PRP$", "WP", "WP$"}
    # Spatial and Temporal deixis are strictly defined by context-dependent wordse lưu code trong đó e dạo ít code theo 
    spatial_words = {"here", "there", "this", "that", "these", "those"}
    temporal_words = {"now", "then", "today", "tomorrow", "yesterday", "ago", "soon", "later"}

    valid_tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            valid_tokens.append(token)

    total_words = len(valid_tokens)

    if total_words == 0:
        return 0.0, 0.0, 0.0

    person_count, spatial_count, temporal_count = 0, 0, 0

    for token in valid_tokens:
        word_lower = token.text.lower()

        if token.tag_ in person_deixis_tags:
            person_count += 1
        if word_lower in spatial_words:
            spatial_count += 1
        if word_lower in temporal_words:
            temporal_count += 1

    return round((person_count/total_words), 5), round((spatial_count/total_words), 5), round((temporal_count/total_words), 5)

# def get_speaking_rate(audio_path:str,
#                         transcript:str="",
#                         ) -> float:
#     """
#     Function to get speaking rate, approximated as number of words divided by total duration.
#     """
#     audio_file = parselmouth.Sound(audio_path)
#     duration = call(audio_file, 'Get end time')
#     word_count = len(str(transcript).split())
#     return word_count / duration if duration > 0 else 0


def pause_count():
    pass

def articulation_rate():
    pass

def pause_rate():
    pass

def syllable_per_word():
    pass

def pause_per_word():
    pass

def pause_per_syllable():
    pass