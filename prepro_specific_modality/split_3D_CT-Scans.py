import json
import re

def count_keywords_in_sentence(keywords, sentence):
    number = sum(1 for keyword in keywords if keyword in sentence.lower())
    if number == 0:
        return sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword), sentence))
    else:
        return number

abdomen_k_w = ['renal', 'bowel', 'kidney', 'liver', 'hepatic', 'lymph', 'mesenteric', 'abdominal', 'colon', 'dilated', 'cysts', 'bladder', 'loops', 'pelvis', 'iliac', 'gallbladder', 'kidneys', 'pole', 'hernia', 'gland', 'portal', 'cyst', 'gas', 'ureter', 'cortical', 'urinary', 'adrenal', 'spleen', 'cystic',
            "hemoperitoneum", "acetabulum", "stomach", "pancrea", "duodenum", "hepat", "stomach", "jejunal", "pudendal", "adnexa", "ileocolic", "colectomy",
            "peritoneal", "abdo", "cecum", "esophagus", "celiac", "vagin", "ureth", "pelvic", "umbilical", "prostate",
            "ileum", "ovar", "rectum", "rectal","uterine", "gallstone", "lumbar", "anal", "rectus", "pelvis", "abdomen"])

head_k_w = ['sinus', 'frontal', 'hemorrhage', 'maxilla', 'cerebral', 'air', 'orbital', 'edema', 'ventricle', 'sinuses', 'hyperdense', 'carotid', 'brain', 'lobe','gland', 'midline', 'muscle', 'intracranial', 'basal', 'acute', 'nasal', 'hematoma', 'extension', 'eye', 'cortical',
            "cerebellum", "vitreous", "iris", "lens", "retina", "basilar", "brain", "cerebellar", "pallidus", "C1", "orbit", "peritonsillar",
            "cisterna magna", "skull", "mandible", "scaphocephaly", "petrous apex", "caudalbasilar artery",
            "fontanelle", "occipital", "sinu", "cranium", "auditory", "foramen", "pterygopalatine", "mastoidectomies", "head",
            "canine", "submandibular", "cranio", "vestibular", "callosum", "lenticulostriate", "cortex", "parietal",
            "cervical", "vocal", "pons", "midbra", "middle ear", "cochlear", "choroid"]

chest_k_w = ['pulmonary', 'lung', 'mediastinal', 'glass', 'ground', 'nodules', 'lymph', 'coronary', 'opacities',
             'lobes', 'arteries', 'aortic', 'lymphadenopathy', 'aorta', 'consolidation', 'chest', 'hilar', 'peripheral',
             'calcified', 'atelectasis', 'subpleural', 'arch', 'pericardial', 'branches', 'dilated', 'air',
             'sided', 'heart',"subclavian", "bronch", "shoulder", "ductus", "diaphragm", "myocardium", "thora", "thymus",
             "epicardi", "thorax", "chest", "rib cage", "sternum", "clavicle", "scapula",
            "Heart", "Lungs", "Trachea", "Esophagus", "Bronchi", "Diaphragm", "Thymus",
            "Aorta", "pulmonary artery", "pulmonary veins", "superior vena cava", "inferior vena cava",
            "Vagus nerve", "phrenic nerve", "Lymph nodes", "thoracic duct", "lobe", "atrial", "sternoclavicular joint", "pneumo", "trachea",
            "left atrium", "breast"]

tibia_k_w = ["femur", "tibia", "limb", "maleol", "cubital"]

with open("ct_quizzes.jsonl", "r", encoding="utf-8") as file:

    with open("chest.jsonl", "w") as c:
        with open("abdomen.jsonl", "w") as a:
            with open("head.jsonl", "w") as b:
                with open("other.jsonl", "w") as au:
                    with open("leg.jsonl", "w") as t:
                        for line in file:
                            data = json.loads(line)
                            text_value = data.get("text", "") 
                            text_value = text_value.split("text:", 1)[-1]

                            tibia_number = count_keywords_in_sentence(tibia_k_w, text_value)
                            abdo_number = count_keywords_in_sentence(abdomen_k_w, text_value)
                            brain_number = count_keywords_in_sentence(head_k_w, text_value)
                            chest_number = count_keywords_in_sentence(chest_k_w, text_value)

                            if abdo_number == 0 and brain_number == 0 and chest_number == 0 and tibia_number == 0:
                                au.write(line)
                            elif abdo_number >= brain_number and abdo_number >= chest_number and abdo_number >= tibia_number:
                                a.write(line)
                            elif brain_number >= chest_number and brain_number >= tibia_number:
                                b.write(line)
                            elif chest_number > tibia_number:
                                c.write(line)
                            else:
                                t.write(line)
