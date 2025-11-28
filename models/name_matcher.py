import json
import difflib

class NameMatcher:
    def __init__(self, names_file):
        with open(names_file, 'r') as f:
            self.names = json.load(f)
    
    def find_best_match(self, input_name):
        best_match = None
        best_score = 0
        for name in self.names:
            score = difflib.SequenceMatcher(None, input_name.lower(), name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = name
        return best_match, best_score
    
    def find_ranked_matches(self, input_name):
        matches = []
        for name in self.names:
            score = difflib.SequenceMatcher(None, input_name.lower(), name.lower()).ratio()
            matches.append((name, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches