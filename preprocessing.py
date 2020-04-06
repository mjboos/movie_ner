import re
import numpy as np
import joblib
memory = joblib.Memory('joblib')

# no apostrophe to remove e.g. "president's" office, no bracket to remove e.g. (V.O.)
character_pattern = r"\b[A-Z]{2}[A-Z\.]+(?: [A-Z\.]{3,})*\b"

character_pattern = r"(?:^|(?:[a-z]+[.!?]\s+))([A-Z-]{2,}[\.\(\)\':;-]*(?:\s+[A-Z-]{2,}[\.\(\)\':;-]*)*)"

stopwords = ['INT', 'EXT', 'CUT', 'BACK', 'END', 'ANGLE', 'CONTINUOUS', "CONT'D", 'RINGS', 'LAUGH', 'DAY', 'NIGHT', 'CONTINUED', 'CAMERA', 'VOICE', 'ROOM', 'OFFICE', 'APARTMENT', 'FALL', 'SPRING', 'WINTER', 'SUMMER', 'LATER', 'MOMENTS LATER', 'HOUSE', 'SAME', 'TIME', 'JUMP', 'CONTINUING']
patterns = [r'\b{}\b'.format(word) for word in stopwords]

def return_character_list(script_string, remove_stopwords=True):
    '''Returns a list of likely character names present in script_string'''
    if remove_stopwords:
        for pattern in patterns:
            script_string = re.sub(pattern, '_stopword_', script_string)

    script_string = re.sub('\t', ' ', script_string)
    script_string = re.sub('\n', ' ', script_string)
    script_string = re.sub(r' +', ' ', script_string)

    matches = [m.group(1).strip('()') for m in re.finditer(character_pattern, script_string)]
    matches, counts = np.unique(matches, return_counts=True)
    return {match: count for match, count in zip(matches, counts)}

def debug_cheap_tokenization(script_string, words_to_keep):
    '''Does very simple tokenization for all words in words_to_keep:
    Returns an N-dimensional ndarray, where N is the number of words in script_string and a dictionary that maps from words to numbers'''
    # replace character names with tokens
    words_to_keep = list(reversed(sorted(words_to_keep, key=len)))
    for word_i, word in enumerate(words_to_keep):
        script_string = re.sub(word, '_TOKEN_{} '.format(word), script_string)
    # TODO: remove a.b patterns
    script_string = re.sub('\t', ' ', script_string)
    script_string = re.sub(r' +', ' ', script_string)
    script_string = script_string.split(' ')
    return script_string

def cheap_tokenization(script_string, words_to_keep):
    '''Does very simple tokenization for all words in words_to_keep:
    Returns an N-dimensional ndarray, where N is the number of words in script_string and a dictionary that maps from words to numbers'''
    # replace character names with tokens
    words_to_keep = list(reversed(sorted(words_to_keep, key=len)))
    for word_i, word in enumerate(words_to_keep):
        script_string = re.sub(word, '_TOKEN_{} '.format(word_i+1), script_string)
    # TODO: remove a.b patterns
    script_string = re.sub('\t', ' ', script_string)
    script_string = re.sub(r' +', ' ', script_string)
    script_string = script_string.split(' ')
    activation = np.zeros(len(script_string))
    for i, word in enumerate(script_string):
        if word.startswith('_TOKEN_'):
            activation[i] = int(word.split('_')[-1])

    word_dict = {word: word_i+1 for word_i, word in enumerate(words_to_keep)}
    inverted_dict = {val: word for word, val in word_dict.items()}
    return activation, word_dict, inverted_dict

def single_activation_to_multi_activation(single_act):
    '''Helper function to go from 1-D activation to multi D activation'''
    multi_act = np.zeros((single_act.shape[0], len(np.unique(single_act))-1))
    for i in range(1, multi_act.shape[1]+1):
        multi_act[single_act==i, i-1] = 1
    return multi_act

def get_often_occurring_characters(char_occ_dict, frequency_threshold=10):
    return [character for character, count in char_occ_dict.items() if count >= frequency_threshold]

def is_duplicate(character, characters):
    return any([(character.startswith(character2) and len(character)==len(character2)+1) for character2 in characters])

def has_part_duplicate(character1, character2):
    if ' ' in character1 and ' ' in character2 and character1 != character2:
        character1 = character1.split(' ')
        character2 = character2.split(' ')
        return character1[0] == character2[0]
    else:
        return False

def load_script_dict():
    import glob
    files = glob.glob('data/*.txt')
    script_dict = {}
    for fl in files:
        with open(fl, 'r') as op:
            script_dict[fl.split('_')[-1][:-4]] = op.read()
    return script_dict

def moving_average_act(multi_act, N=150):
    from scipy.ndimage.filters import uniform_filter1d
    multi_act = uniform_filter1d(multi_act, N, axis=0, origin=N//2-1)
    multi_act[multi_act>1./N] = 1.
    return multi_act

def lp_filter_act(multi_act, N=6, Wn=0.05):
    from scipy.signal import butter, sosfilt
    sos = butter(N, Wn, output='sos')
    # we don't need linear phase
    filtered_mact = np.hstack([sosfilt(sos, act)[:, None] for act in multi_act.T])
    filtered_mact[filtered_mact<0] = 0
    return filtered_mact

def prune_characters(char_occ_dict, threshold=0.1):
    from dirty_cat import SimilarityEncoder
    from sklearn.preprocessing import minmax_scale
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    simenc = SimilarityEncoder(similarity='jaro-winkler')
    transf = simenc.fit_transform(np.array(sorted(char_occ_dict.keys())).reshape(-1, 1))
    corr_dist = minmax_scale(-transf)
    dense_distance = squareform(corr_dist, checks=False)
    Z = linkage(dense_distance, 'average', optimal_ordering=True)
    return get_merged_characters(Z, char_occ_dict, threshold=threshold)

# watch out that everything is sorted 
def get_merged_characters(Z, char_occ_dict, threshold=0.1):
    '''Returns a dictionary that specifies which character names are to be replaced'''
    replace_dict = dict()
    char_occ_copy = {}
    # identify which clusters 
    bolstered_categories = [[cat] for cat in sorted(char_occ_dict.keys())]
    for z_i in Z:
        if z_i[2] < threshold:
            bolstered_categories.append(bolstered_categories[int(z_i[0])]+bolstered_categories[int(z_i[1])])
    for cluster in bolstered_categories[::-1]:
        # get the most common character in cluster
        most_common = sorted([items for items in char_occ_dict.items() if items[0] in cluster], key=lambda x: x[1])[-1][0]
        # sum occurrences for thresholding based on frequency
        cluster_occ_sum = sum([occ for char, occ in char_occ_dict.items() if char in cluster])
        for character in cluster:
            if character not in replace_dict and character != most_common:
                replace_dict[character] = most_common
        if most_common not in char_occ_copy and most_common not in replace_dict:
            char_occ_copy[most_common] = cluster_occ_sum
    return replace_dict, char_occ_copy

def clean_script_string(script_string, replace_dict):
    for to_replace, replace_with in replace_dict.items():
        script_string = re.sub(to_replace, replace_with, script_string)
    return script_string

def process_script(script_string, frequency_threshold=10, char_name_dist_threshold=0.1, **filter_kwargs):
    '''Explain how it works'''
    char_occ_dict = return_character_list(script_string)
    if char_name_dist_threshold:
        replace_dict, char_occ_dict = prune_characters(char_occ_dict, threshold=char_name_dist_threshold)
        script_string = clean_script_string(script_string, replace_dict)

    chars_to_keep = get_often_occurring_characters(char_occ_dict, frequency_threshold=frequency_threshold)
    activation, char_dict, inv_char_dict = cheap_tokenization(script_string, chars_to_keep)
    multi_act = single_activation_to_multi_activation(activation)
    assert multi_act.shape[1] == len(chars_to_keep)
    assert multi_act.shape[1] > 1
    return moving_average_act(multi_act, **filter_kwargs), char_dict, inv_char_dict

@memory.cache
def process_all_scripts():
    script_dict = load_script_dict()
    act_dict = {}
    for script_name, script in script_dict.items():
        try:
            act_dict[script_name] = process_script(script)
        except AssertionError:
            continue
    print('Completed. Errors in {} of {} scripts.'.format(len(script_dict)-len(act_dict),len(script_dict)))
    return act_dict

def process_scripts_debug():
    script_dict = load_script_dict()
    act_dict = {}
    i = 0
    for script_name, script in script_dict.items():
        i += 1
        if i > 100:
            break
        act_dict[script_name] = process_script(script)
    return act_dict



def plot_connectivity(multi_act, characters, func=None):
    from nilearn.plotting import plot_matrix
    if func is None:
        func = np.corrcoef
    corrs = np.corrcoef(multi_act.T)
    return plot_matrix(corrs, labels=characters, reorder=True)

def plot_from_dict(movie, act_dict):
    characters = list(act_dict[movie][1].keys())
    return plot_connectivity(act_dict[movie][0], characters)

if __name__=='__main__':
    act_dict = process_all_scripts()
