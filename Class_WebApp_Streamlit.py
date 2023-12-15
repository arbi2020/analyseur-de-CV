import streamlit as st
import pickle
import re
import nltk
import PyPDF2
import io

nltk.download('punkt')
nltk.download('stopwords')

# Chargement des modèles
model = pickle.load(open('SVM_modele.pkl', 'rb'))
md_vecoriz = pickle.load(open('model_vecoriz.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
def read_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def predict_category_master(resume_text):
    cleaned_resume = clean_resume(resume_text)
    input_features = md_vecoriz.transform([cleaned_resume])
    prediction_id = model.predict(input_features)[0]

    # Mapping des catégories
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    category_name = category_mapping.get(prediction_id, "Inconnu!!!")

    # Paramétrez la recommandation de master en fonction de la catégorie prédite
    recommended_masters = []

    if category_name in ["DevOps Engineer", "Python Developer", "Hadoop", "ETL Developer", "Data Science", "Database", "Business Analyst", "SAP Developer"]:
        recommended_masters.append("Mastère professionnel en Business Analytics & Data Science BADS")

    if category_name in ["Business Analyst", "Data Science", "Database", "DevOps Engineer", "DotNet Developer", "Hadoop", "Java Developer", "Network Security Engineer", "Operations Manager", "Python Developer", "SAP Developer", "Testing", "Web Designing"]:
        recommended_masters.append("Mastère professionnel Ingénierie des Systèmes Emergents (ISEM)")

    if category_name in ["Blockchain", "HR", "Operations Manager", "PMO", "Sales"]:
        recommended_masters.append("Mastère Professionnel en Comptabilité (MPC)")

    if category_name in ["Automation Testing", "Business Analyst", "Data Science", "Database", "DevOps Engineer", "DotNet Developer", "Electrical Engineering", "Java Developer", "Mechanical Engineer", "Network Security Engineer", "Operations Manager", "Python Developer", "SAP Developer", "Web Designing"]:
        recommended_masters.append("Mastère professionnel en Ingénierie avancée des systèmes robotisés et Intelligence artificielle")

    if category_name in ["Blockchain", "HR", "Operations Manager", "PMO", "Sales"]:
        recommended_masters.append("Mastère professionnel en Optimisation et Modernisation de l’Entreprise MOME")

    if category_name in ["Automation Testing", "Business Analyst", "Data Science", "Database", "DevOps Engineer", "DotNet Developer", "Electrical Engineering", "Java Developer", "Mechanical Engineer", "Network Security Engineer", "Operations Manager", "Python Developer", "SAP Developer", "Web Designing"]:
        recommended_masters.append("Mastère professionnel en Nouvelles Technologies des Télécommunications et Réseaux N2TR")

    if category_name in ["DotNet Developer", "Java Developer", "Operations Manager", "Python Developer", "SAP Developer", "Web Designing"]:
        recommended_masters.append("Mastère Professionnel en Ingénierie du Logiciel - Open Source MP2L")

    if category_name in ["Blockchain", "HR", "Operations Manager", "PMO", "Sales"]:
        recommended_masters.append("Mastère professionnel en Management intégré : Qualité, Hygiène, Sécurité et Environnement")

    if category_name in ["Arts", "Health and fitness"]:
        recommended_masters.append("Mastère professionnel en Préparation Physique MP3")

    if category_name in ["Arts", "Health and fitness", "HR", "Blockchain", "Advocate", "PMO", "Sales"]:
        recommended_masters.append("Mastère professionnel en Préparation Mentale M2P2")

    print("Catégorie prédite :", category_name)
    print("Recommandations de master possibles :", recommended_masters)

    return category_name, recommended_masters





def main():
    st.title("ANALYSE DE CV ET RECOMMENDATION DE MASTERE A L UNIVERSITE VIRTUELLE DE TUNIS ")
    uploaded_file = st.file_uploader('Téléchargez le CV', type=['pdf'])

    if uploaded_file is not None:
        resume_text = read_pdf(uploaded_file)
        category_name, recommended_masters = predict_category_master(resume_text)

        st.write("Catégorie de CV prédite : ", category_name)

        if recommended_masters:
            st.write("Recommandations de master possibles :")
            for master in recommended_masters:
                st.text(master)
        else:
            st.write("Aucune recommandation de master disponible pour cette catégorie.")

if __name__ == "__main__":
    main()


