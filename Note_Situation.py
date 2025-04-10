import pandas as pd
import json
import requests
import torch
import smtplib
import pytz
import unicodedata
import fitz  # PyMuPDF
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fpdf import FPDF
from openai import OpenAI
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from secret import OPENAI_API_KEY
torch.serialization.add_safe_globals([XttsConfig])
import schedule
import time

# ============= CONFIGURATION GLOBALE =============
TOKEN = "HSv2mYkd7MbZ2rkiVTqE3Gh3AAiMh9SA"
EMAIL_EXPEDITEUR = "salim.mansouri15@gmail.com"
EMAIL_PASSWORD = "fqum hxmq agup saks"
TIMEZONE = "Europe/Paris"

# ============= FONCTIONS DE COMMUNICATION =============
def envoyer_pdf_par_mail(pdf_file_path, audio_file_path, destinataire, sujet="Synthèse perturbations", message="Veuillez trouver en pièce jointe le rapport PDF."):
    """
    Envoie un email contenant deux pièces jointes : un fichier PDF et un fichier audio.
    
    Args:
        pdf_file_path (str): Chemin du fichier PDF à envoyer.
        audio_file_path (str): Chemin du fichier audio à envoyer.
        destinataire (str): Adresse email du destinataire.
        sujet (str): Sujet du mail.
        message (str): Corps du message.
    """
    # Création du message
    msg = MIMEMultipart()
    msg['From'] = EMAIL_EXPEDITEUR
    msg['To'] = destinataire
    msg['Subject'] = sujet

    # Corps du message
    msg.attach(MIMEText(message, 'plain'))

    with open(pdf_file_path, 'rb') as f:
        part = MIMEApplication(f.read(), Name=pdf_file_path)
        part['Content-Disposition'] = f'attachment; filename="{pdf_file_path}"'
        msg.attach(part)

    with open(audio_file_path, 'rb') as f:
        audio_part = MIMEApplication(f.read(), Name=audio_file_path)
        audio_part['Content-Disposition'] = f'attachment; filename="{(audio_file_path)}"'
        msg.attach(audio_part)

    # Connexion SMTP sécurisée (Gmail)
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_EXPEDITEUR, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Mail envoyé à {destinataire}")
    except Exception as e:
        print(f"Échec de l'envoi : {e}")

def call_siri_api():
    """
    Appelle l'API SIRI pour récupérer les messages généraux
    
    Returns:
        str: Données JSON de l'API ou None en cas d'erreur
    """
    url = "https://prim.iledefrance-mobilites.fr/marketplace/general-message?LineRef=ALL"
    
    try:
        headers = {
            "apiKey": TOKEN,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Erreur lors de l'appel API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception lors de l'appel API: {e}")
        return None

# ============= FONCTIONS DE TRAITEMENT DE DONNÉES =============
def extraire_contenu_pdf_modele(fichier_pdf_exemple):
    """
    Extrait le contenu textuel d'un fichier PDF exemple
    
    Args:
        fichier_pdf_exemple (str): Chemin du fichier PDF à extraire
        
    Returns:
        str: Contenu textuel du PDF
    """
    doc = fitz.open(fichier_pdf_exemple)
    contenu = ""
    for page in doc:
        contenu += page.get_text()
    return contenu.strip()

def extract_siri_data(json_data):
    """
    Extrait et filtre les informations pertinentes du message SIRI
    Filtre seulement les messages créés dans les 15 dernières minutes
    
    Args:
        json_data (str): Données JSON de l'API SIRI
        
    Returns:
        DataFrame: Données extraites et filtrées
    """
    # Charger les données JSON
    data = json.loads(json_data)
    
    # Vérifier si les données ont le format attendu
    if not data.get('Siri') or not data['Siri'].get('ServiceDelivery') or not data['Siri']['ServiceDelivery'].get('GeneralMessageDelivery'):
        print("Le format des données SIRI n'est pas celui attendu.")
        return pd.DataFrame()
    
    # Liste pour stocker les données extraites
    rows = []
    
    # Calculer le timestamp d'il y a 15 minutes
    now = datetime.now(pytz.timezone(TIMEZONE))
    fifteen_minutes_ago = now - timedelta(minutes=15)

    print("Heure actuelle:", now)
    print("15 min avant:", fifteen_minutes_ago)
    
    # Parcourir les messages
    for message_delivery in data['Siri']['ServiceDelivery']['GeneralMessageDelivery']:
        if 'InfoMessage' in message_delivery:
            for info in message_delivery['InfoMessage']:
                # Extraire le timestamp d'enregistrement
                recorded_time_str = info.get('RecordedAtTime', '')
                
                # Vérifier si le message est récent (moins de 15 minutes)
                if recorded_time_str:
                    try:
                        # Convertir le timestamp ISO en objet datetime
                        recorded_time = datetime.fromisoformat(recorded_time_str.replace('Z', '+00:00'))
                        
                        # Skip si le message est plus ancien que 15 minutes
                        if recorded_time < fifteen_minutes_ago:
                            continue
                            
                    except (ValueError, TypeError) as e:
                        print(f"Erreur lors du parsing de la date: {e}, date: {recorded_time}")
                        continue
                
                # Extraire les informations nécessaires
                valid_until = info.get('ValidUntilTime', '')
                
                line_ref = "Non spécifiée"
                # Extraire la ligne concernée (si disponible)
                if 'Content' in info and 'LineRef' in info['Content'] and info['Content']['LineRef']:
                    line_ref = info['Content']['LineRef'][0].get('value', '')

                    if 'STIF:Line::' in line_ref:
                        line_ref = line_ref.split('STIF:Line::')[1].strip(':')
                    
                # Catégorie (type de message)
                categorie = info.get('InfoChannelRef', {}).get('value')
                
                # Texte du message
                texte = ""
                if 'Content' in info and 'Message' in info['Content'] and info['Content']['Message']:
                    message_text = info['Content']['Message'][0].get('MessageText', {})
                    if message_text:
                        texte = message_text.get('value', '')
                
                # Convertir les dates ISO en format plus lisible
                try:
                    heure_locale = recorded_time.astimezone(ZoneInfo(TIMEZONE))
                    date_debut = heure_locale.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    date_debut = recorded_time_str
                
                try:
                    date_fin = datetime.fromisoformat(valid_until.replace('Z', '+00:00')).astimezone(ZoneInfo(TIMEZONE)).strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    date_fin = valid_until
                
                # Ajouter à notre liste de données
                rows.append({
                    "Lignes": line_ref,
                    "Date de début": date_debut,
                    "Date de fin": date_fin,
                    "Catégorie": categorie,
                    "Texte de l'ICV gare et bord": texte
                })
    
    # Créer un dataframe avec nos données
    df = pd.DataFrame(rows)
    if df.empty:
        print("Aucun message créé dans les 15 dernières minutes n'a été trouvé.")
    else:
        print(f"Nombre de messages récents trouvés: {len(df)}")
    
    return df

def generate_prompt(dataframe, modele_pdf):
    """
    Génère un prompt pour l'API OpenAI à partir des données extraites
    
    Args:
        dataframe (DataFrame): Données extraites et filtrées
        modele_pdf (str): Contenu du PDF exemple
        
    Returns:
        str: Prompt formaté pour l'API OpenAI
    """
    prompt = (
        "Tu es un assistant de la SNCF. À partir des données suivantes extraites de messages SIRI "
        "des 15 dernières minutes, rédige une synthèse claire, concise et structurée à destination du directeur opérationnel.\n\n"
    )

    prompt += "Utilise comme **référence de ton style** la synthèse suivante extraite d'un PDF précédent :\n\n"
    prompt += modele_pdf + "\n\n"

    prompt += (
        "Structure ta synthèse de la façon suivante :\n"
        "- Réseau **Train-RER**\n"
        "- Réseau **Métro**\n"
        "- Réseau **Tram-Train**\n"

        "Si une **ligne** ou un **mode de transport** est marqué comme *Non spécifiée*, "
        "essaye de le **déduire à partir du texte du message** (gares citées, tronçons, noms de ligne, RER, etc.) sinon n'en parle pas. "
        "Classe l'information dans le bon réseau si la déduction est fiable.\n\n"

        "Voici les données brutes à utiliser :\n\n"
    )

    # Regrouper par type de réseau
    mapping = {
        "rail": "Train-RER",
        "metro": "Métro",
        "tram": "Tram-Train",
        "non_specifie": "Non spécifiée"
    }

    # S'assurer que toutes les colonnes nécessaires existent
    if 'TransportMode' not in dataframe.columns:
        dataframe['TransportMode'] = 'non_specifie'

    for mode_technique in ['rail', 'metro', 'tram', 'non_specifie']:
        bloc_df = dataframe[dataframe['TransportMode'] == mode_technique]
        if bloc_df.empty:
            continue
        prompt += f"--- Réseau {mapping[mode_technique]} ---\n"
        for _, row in bloc_df.iterrows():
            texte = row["Texte de l'ICV gare et bord"].strip().replace('\n', ' ')
            prompt += (
                f"Ligne {row['Name_Line']} | {row['Catégorie']} | "
                f"{row['Date de début']} - {row['Date de fin']} | {texte}\n"
            )
        prompt += "\n"

    prompt += (
        "\nRédige maintenant la synthèse structurée comme indiqué, suivie de l'analyse des impacts inter-lignes.\n"
        "N'invente rien. Tu peux regrouper les perturbations similaires.\n"
    )

    return prompt

def enregistrer_pdf(nom_fichier, contenu_texte):
    """
    Enregistre un contenu texte dans un fichier PDF
    
    Args:
        nom_fichier (str): Nom du fichier PDF à créer
        contenu_texte (str): Contenu à enregistrer dans le PDF
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # Remplacer les caractères problématiques
    replacements = {
        '\u2022': '-', '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '--', '\u2026': '...',
        '\u2006': ' ',  # espace fine
    }

    for old, new in replacements.items():
        contenu_texte = contenu_texte.replace(old, new)

    # Supprimer ou normaliser les caractères non Latin-1 restants
    def to_latin1(text):
        return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')

    contenu_texte = to_latin1(contenu_texte)

    for ligne_contenu in contenu_texte.split('\n'):
        pdf.multi_cell(0, 10, ligne_contenu)

    pdf.output(nom_fichier)

# ============= FONCTIONS PRINCIPALES =============
def process_siri_to_pdf():
    """
    Processus principal: appelle l'API SIRI, traite les données et génère un PDF
    
    Returns:
        str: Nom du fichier PDF généré ou message d'erreur
    """
    # Charger le PDF modèle et le référentiel des lignes
    modele_pdf = extraire_contenu_pdf_modele("Note de situation.pdf")
    
    df_ligne = pd.read_csv("referentiel-des-lignes.csv", delimiter=";")
    # Filtrage des bus
    df_ligne = df_ligne[df_ligne['TransportMode'] != 'bus']
    df_ligne = df_ligne[['ID_Line', 'Name_Line', 'TransportMode']]
    
    # Appeler l'API SIRI
    print("Appel de l'API SIRI...")
    siri_json_data = call_siri_api()
    
    if not siri_json_data:
        print("Échec de l'appel API.")
        return "Échec de l'appel API - Aucun PDF généré"
    
    print("Données reçues de l'API SIRI.")
    
    # Extraire les données du JSON SIRI (uniquement les 15 dernières minutes)
    df = extract_siri_data(siri_json_data)

    # Vérifier si le dataframe est vide
    if df.empty:
        print("Aucun message récent trouvé. Traitement arrêté.")
        return "Pas de nouveau PDF généré - aucune perturbation récente."

    print("\nDF API :")
    print(df)

    print("\nDF référentiel :")
    print(df_ligne.head())

    # Joindre les données avec le référentiel des lignes
    final_df = df.merge(df_ligne, how='left', left_on='Lignes', right_on='ID_Line')

    # Remplacer NaN par "Non spécifiée" dans Name_Line uniquement si Lignes vaut "Non spécifiée"
    final_df['Name_Line'] = final_df.apply(
        lambda row: "Non spécifiée" if row['Lignes'] == "Non spécifiée" else row['Name_Line'], 
        axis=1
    )

    final_df['TransportMode'] = final_df.apply(
        lambda row: "non_specifie" if row['Lignes'] == "Non spécifiée" else row['TransportMode'], 
        axis=1
    )

    # Supprimer les lignes dont l'ID de ligne n'est pas reconnu
    final_df = final_df[(final_df['Name_Line'].notna()) | (final_df['Name_Line'] == "Non spécifiée")].copy()

    # Supprimer l'ID
    final_df.drop(columns=['ID_Line'], inplace=True)

    print("\nDF finale :")
    print(final_df.head(), "\n")
    
    if final_df.empty:
        print("Aucune donnée pertinente n'a été extraite du message SIRI (15 dernières minutes).")
        return "Pas de nouveau PDF généré - données non pertinentes"
        
    print(f"Données extraites: {len(final_df)} messages des 15 dernières minutes")
    
    # 5. Générer un prompt pour OpenAI
    prompt_text = generate_prompt(final_df, modele_pdf)
    print("Prompt généré pour OpenAI")
    
    # 6. Appel à l'API OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en rédaction opérationnelle pour la SNCF."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.4,
        max_tokens=1200
    )

    synthese_text = response.choices[0].message.content
    print("Synthèse générée par OpenAI")
    
    # 7. Génération du PDF
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_pdf_name = f"Synthese_Perturbations_{timestamp}.pdf"
        enregistrer_pdf(output_pdf_name, synthese_text)
        print(f"PDF généré avec succès: {output_pdf_name}")
        return output_pdf_name
    except Exception as e:
        print(f"Erreur lors de la génération du PDF: {e}")
        return f"Erreur lors de la génération du PDF: {e}"

def tts(pdf_file):
    """
    Génère un fichier audio à partir du contenu d'un fichier texte
    
    Args:
        file (str): Chemin du fichier texte à convertir en audio
    """
    now = datetime.now(pytz.timezone(TIMEZONE))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    txt_content = extraire_contenu_pdf_modele(pdf_file)

    print("Génartion de l'audio")
    output_file = f"output_{now}.wav"
    tts.tts_to_file(
        text=txt_content, 
        speaker_wav="my/cloning/les-lectures-de-simone-un-nouveau-pneu.wav", 
        language="fr", 
        file_path=output_file,
        speed=0.9
    )
    print(f"Fichier audio généré: {output_file}")

    return output_file

def job():
    """
    Fonction principale qui sera exécutée toutes les 15 minutes
    """
    print(f"\n[{datetime.now()}] EXÉCUTION PLANIFIÉE DÉMARRÉE\n")
    try:
        # Exécuter le processus principal
        pdf_file = process_siri_to_pdf()
        print(f"Traitement terminé. Fichier PDF: {pdf_file}")
        
        # Si un PDF a été généré
        if pdf_file and not pdf_file.startswith("Pas de") and not pdf_file.startswith("Erreur"):
            # Sujet avec date et heure locale
            heure_point = datetime.now(pytz.timezone(TIMEZONE))
            sujet_automatique = heure_point.strftime("Point de situation - %d/%m/%Y à %Hh%M")

            # Génération audio
            audio_file = tts(pdf_file)
            
            # Envoi par mail
            envoyer_pdf_par_mail(
                pdf_file_path=pdf_file,
                audio_file_path=audio_file,
                destinataire=EMAIL_EXPEDITEUR,
                sujet=sujet_automatique,
                message="Bonjour,\n\nVoici la synthèse des perturbations SNCF des 15 dernières minutes.\n\nCordialement."
            )
        else:
            print("Aucun PDF généré ou erreur lors de la génération. Aucun email envoyé.")
    except Exception as e:
        print(f"Erreur lors du traitement planifié: {e}")
    print(f"[{datetime.now()}] EXÉCUTION PLANIFIÉE TERMINÉE\n")

# ============= POINT D'ENTRÉE =============
if __name__ == "__main__":
    print(f"Démarrage du service de surveillance des perturbations ({datetime.now()})")
    print(f"Le script va maintenant s'exécuter toutes les 15 minutes.")
    
    # Exécuter une première fois au démarrage
    job()
    
    # Planifier l'exécution toutes les 15 minutes
    schedule.every(15).minutes.do(job)
    
    # Boucle principale pour maintenir le script en exécution
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScript arrêté manuellement.")
    except Exception as e:
        print(f"Erreur dans la boucle principale: {e}")