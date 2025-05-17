'''pip install numpy
pip install scipy
pip istall scikit-learn
pip install matplotlib
pip install -U spacy
pip install spacy_transformers
pip install fasttext
'''

import spacy
import os
import tika
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#import fasttext
import re
import joblib
from pathlib import Path

# Load models
resume_nlp = spacy.load ('ml_model/resume_train/model-best')
jd_nlp = spacy.load ('ml_model/jd_train/model-best')
we_model = fasttext.load_model('/content/drive/MyDrive/fasttext_we/cc.en.300.bin')


# Import Module
import os
import tika

resume_path = r'/content/drive/MyDrive/test/resume/'
jd_path = r'/content/drive/MyDrive/test/jd/'
string_text = ""

resume_labels = ['EMAIL ADDRESS', 'WORKED AS', 'DEGREE', 'CERTIFICATION', 'COMPANIES WORKED AT', 'LANGUAGE', 'LOCATION', 'LINKEDIN LINK', 'YEAR OF GRADUATION', 'COLLEGE NAME', 'NAME', 'SKILLS', 'YEARS OF EXPERIENCE', 'UNIVERSITY', 'AWARDS']
jd_labels = ['JOB TITLE', 'COMPANY NAME', 'COMPANY LOCATION', 'CATEGORY', 'TYPE OF JOB', 'SKILLS', 'EDUCATION', 'LANGUAGE', 'EXPERIENCE', 'SALARY', 'RESPONSIBILITIES', 'GENDER', 'LEVEL']

resume_dicts = []
jd_dicts = []

# Read text File
def read_text_file(file_path):
    with open(file_path, 'rb') as file_obj:
        response = tika.parser.from_file(file_obj)

    # str_text += str(response['content']) + '\n\n' # Add all text into a str_text
    string_text = str(response['content']).lstrip('\n')
    head_tail = os.path.split(file_path)
    file_name = head_tail[1].split('.pdf')[0]
    print(file_name)
    file_newpath = head_tail[0] + '/' + file_name + '.txt'
    print(file_newpath)

    if os.path.exists(file_newpath) == False:
        file = open(file_newpath, 'x')
        file.close()

    with open(file_newpath, 'w') as file:
        file.write(string_text)

    return file_newpath

# Iterate through all files in the resume folder
for file in os.listdir(resume_path):

    # Check whether the file is in text format or not
    if file.endswith(".pdf"):
        current_dict = {}
        file_path = f"{resume_path}/{file}"

        # Text extraction
        file_newpath = read_text_file(file_path)
        with open(file_newpath, 'r') as file:
          data = file.read()

        # NER
        ner_data = resume_nlp(data)

        for ent in ner_data.ents:
          if ent.label_ in resume_labels:
            if ent.label_ not in current_dict:
              current_dict[ent.label_] = [ent.text]
              # print(ent.text)
            else:
              current_dict[ent.label_].extend([ent.text])
              # print(ent.text)
          else:
            print(f"Missing:{ent.label_}")

        # Resume dictionary
        resume_dicts.append(current_dict)

print(resume_dicts)


# Iterate through all files in the jd folder
for file in os.listdir(jd_path):
    current_dict = {}
    # Check whether the file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{jd_path}/{file}"

        with open(file_path, 'r') as file:
          data = file.read()

        ner_data = jd_nlp(data)

        for ent in ner_data.ents:
          if ent.label_ in jd_labels:
            if ent.label_ not in current_dict:
              current_dict[ent.label_] = [ent.text]
            else:
              current_dict[ent.label_].extend([ent.text])
          else:
            print(f"Missing:{ent.label_}")

        # Resume dictionary
        jd_dicts.append(current_dict)

print(jd_dicts)

def feature_vector(features):
    feature_vectors = []

    for feature in features:
      for word in feature:
        word_vec = []
        word_vec.append(np.mean(we_model.get_word_vector(word)))
      # print(word_vec)
      feature_vectors.append(word_vec)
    return feature_vectors

jds = jd_dicts[0]

jd_features = {}
jd_features["EXPERIENCE"] = {}

if "SKILLS" in jds or "JOB TITLE" in jds:
    jd_features["SKILLS"] = jds["SKILLS"] + jds["JOB TITLE"]

for lists in jds["EXPERIENCE"]:
    match = re.search(r"(\d+)\s*years", lists)
    if match:
      jd_features["EXPERIENCE"]["YEARS OF EXPERIENCE"] = int(match.group(1))
    else:
      jd_features["SKILLS"].append(lists)

if "LANGUAGE" in jds:
    jd_features["LANGUAGE"] = jds["LANGUAGE"]

if "EDUCATION" in jds:
    jd_features["EDUCATION"] = jds["EDUCATION"]

if any(key in jd_dict for key in ["JOB TITLE", "CATEGORY", "EXPERIENCE"]):
    jd_features["EXPERIENCE"]["TITLE"] = jds["JOB TITLE"] + jds["CATEGORY"]

# print(jd_features)

def list_features(resume_dict, resume_features):
  for key in resume_dict.keys():
    if key == "SKILLS":
        resume_features[key].extend(resume_dict[key])
    elif key == "LANGUAGE":
        resume_features[key].extend(resume_dict[key])
    elif key == "DEGREE":
        resume_features["EDUCATION"].extend(resume_dict[key])
    elif key == "CERTIFICATION":
        resume_features["EDUCATION"].extend(resume_dict[key])
    elif key == "YEARS OF EXPERIENCE":
      for experience_years in resume_dict[key]:
          # print(experience_years)
          match = re.search(r"(\d*)\s*(?:years?)??\s*(\d*)\s*months?", str(experience_years))
          if match:
            if (match.group(1) and match.group(2)):
                years = int(match.group(1))
                months = int(match.group(2))
            else:
                years = 0
                months = int(match.group(1))
            total_experience = years + (months / 12)
            resume_features["EXPERIENCE"]["YEARS OF EXPERIENCE"].append("{:.2f}".format(total_experience))
    elif key == "WORKED AS":
        resume_features["EXPERIENCE"]["TITLE"] = resume_dict["WORKED AS"]
  # print(resume_features)
  return resume_features


vector_lists = []
resume_features = {}


for resume_dict, jd_dict in zip(resume_dicts, [jds]*len(resume_dicts)):
    resume_features = {"EXPERIENCE": {"TITLE":[], "YEARS OF EXPERIENCE":[]}, "SKILLS": [], "EDUCATION": [], "LANGUAGE": []}
    vec = {}

    resume_features = list_features(resume_dict, resume_features)

    # Check if title and year length match
    if len(resume_features["EXPERIENCE"]["TITLE"]) == len(resume_features["EXPERIENCE"]["YEARS OF EXPERIENCE"]):
      # print("Equal")
      pass
    else:
      resume_features["EXPERIENCE"]["YEARS OF EXPERIENCE"]= resume_features["EXPERIENCE"]["YEARS OF EXPERIENCE"][:len(resume_features["EXPERIENCE"]["TITLE"])]

    # print(resume_features["EXPERIENCE"])

    for resumes, jd in zip(resume_features.keys(), jd_features.keys()):
      if resumes == "EXPERIENCE":
        for resume_years, resume_title in zip(resume_features["EXPERIENCE"]["YEARS OF EXPERIENCE"],resume_features["EXPERIENCE"]["TITLE"]):
          if resume_title in jd_features["EXPERIENCE"]["TITLE"]:
            val = (float(jd_features["EXPERIENCE"]["YEARS OF EXPERIENCE"]) - float(resume_years))
            # print(val)
            if val >= 0:
              vec[resumes] = val
            else:
              vec[resumes] = 0
        # print(vec[resumes])
      if resumes in ["SKILLS", "EDUCATION", "LANGUAGE"]:
        # print(f"{resumes}:",resume_features[resumes])
        if resume_features[resumes]:
          res_vec = np.concatenate(np.array(feature_vector(resume_features[resumes])))
          # print(f"{resumes}:",(res_vec))
          jd_vec = np.concatenate(np.array(feature_vector(jd_features[jd])))
          # print(f"{resumes}:",jd_vec)

          if len(res_vec) != len(jd_vec):
            # print("Warning: Feature vectors have different dimensions.")

            # Pad the shorter vector with zeros
            max_length = max(len(res_vec), len(jd_vec))
            res_vec = np.concatenate([res_vec, np.zeros(max_length - len(res_vec))])
            jd_vec = np.concatenate([jd_vec, np.zeros(max_length - len(jd_vec))])

            # print(res_vec)
            # print(jd_vec)

            similarity_score = np.mean(cosine_similarity(res_vec.reshape(1, -1), jd_vec.reshape(1, -1)))

            vec[resumes] = round(similarity_score,3)

    vector_lists.append(vec)
    print(vec)

print(vector_lists)

#########EDITTTTTTTTTTTTTTTTTTT
# weights
weight = {"EXPERIENCE":30, "SKILLS": 40, "EDUCATION": 20, "LANGUAGE": 10}

score = []
for i,vectors in enumerate(vector_lists):
  total_weight = 0.0
  for score_key,weight_key in zip(vectors.keys(),weight.keys()):
    total_weight += vectors[score_key]*weight[weight_key]
  score.append(round(total_weight, 3))
  resume_dicts[i]["Score"] = round(total_weight, 3)
  print(resume_dicts[i])

print(score)
print(resume_dicts)