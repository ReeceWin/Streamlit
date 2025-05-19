import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from tqdm import tqdm

class NHSConditionScraper:
    """
    Scraper for NHS Conditions Database
    Extracts treatment advice and symptoms for medical conditions
    """
    
    def __init__(self, output_file="nhs_conditions_database.json"):
        self.output_file = output_file
        self.base_url = "https://www.nhs.uk/conditions/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        # Common name mapping between dataset names and NHS website URLs
        self.name_mapping = {
            "Fungal infection": "fungal-nail-infection",  # Most common type
            "GERD": "heartburn-and-acid-reflux",  # GERD is called acid reflux in NHS
            "(vertigo) Paroymsal Positional Vertigo": "vertigo",
            "Chronic cholestasis": "primary-biliary-cholangitis",  # Related condition
            "Peptic ulcer diseae": "stomach-ulcer",  # Corrected spelling and NHS term
            "Dimorphic hemmorhoids(piles)": "piles-haemorrhoids",
            "Gastroenteritis": "diarrhoea-and-vomiting",  # NHS combines these symptoms
            "Bronchial Asthma": "asthma",
            "Hypertension ": "high-blood-pressure-hypertension",
            "Cervical spondylosis": "cervical-spondylosis",
            "Paralysis (brain hemorrhage)": "brain-haemorrhage",
            "Jaundice": "jaundice",
            "Malaria": "malaria",
            "Chicken pox": "chickenpox",
            "Dengue": "dengue",
            "Typhoid": "typhoid-fever",
            "Hepatitis A": "hepatitis-a",
            "Hepatitis B": "hepatitis-b",
            "Hepatitis C": "hepatitis-c",
            "Hepatitis D": "hepatitis",  # General page, as D is rare
            "Hepatitis E": "hepatitis-e",
            "Alcoholic hepatitis": "alcohol-related-liver-disease",
            "Tuberculosis": "tuberculosis-tb",
            "Common Cold": "common-cold",
            "Pneumonia": "pneumonia",
            "Urinary tract infection": "urinary-tract-infections-utis",
            "Psoriasis": "psoriasis",
            "Impetigo": "impetigo",
            "Hyperthyroidism": "overactive-thyroid-hyperthyroidism",
            "Hypoglycemia": "low-blood-sugar-hypoglycaemia",
            "Hypothyroidism": "underactive-thyroid-hypothyroidism",
            "Osteoarthristis": "osteoarthritis",
            "Arthritis": "rheumatoid-arthritis"  # Most common form
        }
        
        # These need manual entries as they don't have direct NHS pages
        self.manual_entries = {
            "AIDS": {
                "name": "AIDS",
                "symptoms": [
                    "Recurring fever",
                    "Extreme and unexplained tiredness",
                    "Rapid weight loss",
                    "Muscle wasting",
                    "Severe night sweats",
                    "Prolonged swelling of lymph glands",
                    "Chronic diarrhea",
                    "Patches in throat"
                ],
                "advice": [
                    "AIDS requires specialized medical treatment with antiretroviral medication",
                    "Consult a healthcare provider immediately for proper testing and treatment",
                    "Treatment can help manage HIV infection and prevent progression to AIDS",
                    "Support groups can provide emotional and practical assistance",
                    "Regular medical check-ups are essential to monitor the condition"
                ],
                "source": "Manual entry based on NHS guidelines",
                "url": "https://www.nhs.uk/conditions/hiv-and-aids/"
            },
            "Varicose veins": {
                "name": "Varicose veins",
                "symptoms": [
                    "Twisted and enlarged veins, usually on legs",
                    "Blue or dark purple appearance",
                    "Swollen, raised veins",
                    "Aching legs",
                    "Burning or throbbing in legs",
                    "Muscle cramp in legs",
                    "Dry, itchy skin over affected veins"
                ],
                "advice": [
                    "Exercise regularly to improve circulation",
                    "Avoid standing or sitting for long periods",
                    "Elevate your legs when resting",
                    "Wear compression stockings to help blood flow",
                    "Consult a doctor if varicose veins are painful or causing discomfort",
                    "Medical treatments include endothermal ablation, sclerotherapy, or surgery in severe cases"
                ],
                "source": "Manual entry based on NHS guidelines",
                "url": "https://www.nhs.uk/conditions/varicose-veins/"
            },
            "Acne": {
                "name": "Acne",
                "symptoms": [
                    "Greasy skin",
                    "Spots and blackheads",
                    "Skin inflammation",
                    "Scarring"
                ],
                "advice": [
                    "Wash affected areas with mild soap or cleanser and lukewarm water",
                    "Don't wash too often as it can irritate skin",
                    "Use oil-free or water-based products on your face",
                    "Avoid squeezing spots as it can cause scarring",
                    "Over-the-counter treatments containing benzoyl peroxide may help",
                    "If acne is severe or affecting your self-esteem, consult a GP"
                ],
                "source": "Manual entry based on NHS guidelines",
                "url": "https://www.nhs.uk/conditions/acne/"
            },
            "Migraine": {
                "name": "Migraine",
                "symptoms": [
                    "Moderate to severe headache",
                    "Pain on one side of the head",
                    "Pulsating or throbbing pain",
                    "Nausea and vomiting",
                    "Sensitivity to light and sound",
                    "Visual disturbances"
                ],
                "advice": [
                    "Rest in a quiet, dark room when experiencing a migraine",
                    "Apply a cold pack to your head or neck",
                    "Take over-the-counter pain medications like aspirin, ibuprofen, or acetaminophen",
                    "Try to maintain a regular sleep pattern",
                    "Keep a migraine diary to identify triggers",
                    "Consult a doctor if migraines are severe or frequent"
                ],
                "source": "Manual entry based on NHS guidelines",
                "url": "https://www.nhs.uk/conditions/migraine/"
            },
            "Drug Reaction": {
                "name": "Drug Reaction",
                "symptoms": [
                    "Skin rashes",
                    "Itching",
                    "Fever",
                    "Swelling",
                    "Breathing difficulties",
                    "Stomach pain"
                ],
                "advice": [
                    "Stop taking the medication if you suspect a drug reaction (unless advised otherwise by a doctor)",
                    "Seek immediate medical attention for severe reactions such as difficulty breathing or severe rash",
                    "For mild reactions, antihistamines may help relieve symptoms",
                    "Always inform healthcare providers about any previous drug reactions",
                    "Wear a medical ID bracelet if you have serious drug allergies"
                ],
                "source": "Manual entry based on NHS guidelines",
                "url": "https://www.nhs.uk/conditions/allergies/"
            }
        }
        
    def format_condition_name(self, condition_name):
        """Format condition name for URL"""
        # Check if we have a mapping
        if condition_name in self.name_mapping:
            return self.name_mapping[condition_name]
            
        # Otherwise format it according to NHS conventions
        formatted_name = condition_name.lower()
        formatted_name = re.sub(r'\s+', '-', formatted_name)  # Replace spaces with hyphens
        formatted_name = re.sub(r'[^\w\-]', '', formatted_name)  # Remove special characters
        return formatted_name
    
    def scrape_condition(self, condition_name):
        """Scrape condition information from NHS website"""
        # Check if we have a manual entry
        if condition_name in self.manual_entries:
            print(f"Using manual entry for {condition_name}")
            return self.manual_entries[condition_name]
            
        # Format condition for URL
        formatted_name = self.format_condition_name(condition_name)
        url = f"{self.base_url}{formatted_name}/"
        
        # Add delays between requests to be respectful
        time.sleep(2)
        
        try:
            # Get the webpage
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to retrieve {condition_name} ({url}): {response.status_code}")
                return self.create_fallback_entry(condition_name)
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract data
            result = {
                "name": condition_name,
                "symptoms": self.extract_symptoms(soup),
                "advice": self.extract_advice(soup),
                "source": "NHS UK Conditions Database",
                "url": url
            }
            
            # If we didn't find any advice, try alternative sections
            if not result["advice"]:
                result["advice"] = self.extract_alternative_advice(soup)
            
            # If still no advice, create a fallback entry
            if not result["advice"]:
                return self.create_fallback_entry(condition_name)
                
            return result
            
        except Exception as e:
            print(f"Error scraping {condition_name}: {str(e)}")
            return self.create_fallback_entry(condition_name)
    
    def extract_symptoms(self, soup):
        """Extract symptoms from NHS page"""
        symptoms = []
        
        # Try to find symptoms section
        symptoms_section = soup.find('h2', string=re.compile(r'Symptoms|Signs and symptoms'))
        
        if symptoms_section:
            # Find lists after the symptoms heading
            ul_elements = symptoms_section.find_next_siblings('ul')
            for ul in ul_elements:
                if ul_elements.index(ul) > 2:  # Don't go too far from the heading
                    break
                for li in ul.find_all('li'):
                    if li.text.strip():
                        symptoms.append(li.text.strip())
            
            # If no list found, try paragraphs
            if not symptoms:
                next_elem = symptoms_section.find_next()
                while next_elem and next_elem.name != 'h2':
                    if next_elem.name == 'p' and next_elem.text.strip():
                        symptoms.append(next_elem.text.strip())
                    next_elem = next_elem.find_next()
        
        return symptoms
    
    def extract_advice(self, soup):
        """Extract treatment advice from NHS page"""
        advice = []
        
        # Find the treatment section
        treatment_section = soup.find('h2', string=re.compile(r'Treatment|How to treat|Managing|Self-help'))
        
        if treatment_section:
            # Check for lists first
            ul_elements = treatment_section.find_next_siblings('ul')
            for ul in ul_elements:
                if ul_elements.index(ul) > 2:  # Don't go too far from the heading
                    break
                for li in ul.find_all('li'):
                    if li.text.strip():
                        advice.append(li.text.strip())
            
            # If no list, get paragraphs
            if not advice:
                next_elem = treatment_section.find_next()
                count = 0
                while next_elem and next_elem.name != 'h2' and count < 5:
                    if next_elem.name == 'p' and next_elem.text.strip():
                        advice.append(next_elem.text.strip())
                        count += 1
                    next_elem = next_elem.find_next()
        
        return advice
    
    def extract_alternative_advice(self, soup):
        """Try to extract advice from other sections if treatment section not found"""
        advice = []
        
        # Try other common headings
        alternative_headings = [
            re.compile(r'What you can do|Self-care|Home treatment|Manage|Advice'),
            re.compile(r'Prevention|How to prevent|Preventing'),
            re.compile(r'When to see a doctor|When to seek help')
        ]
        
        for pattern in alternative_headings:
            section = soup.find(['h2', 'h3'], string=pattern)
            if section:
                # Check for lists
                for ul in section.find_next_siblings('ul'):
                    for li in ul.find_all('li'):
                        if li.text.strip():
                            advice.append(li.text.strip())
                
                # If no list, get paragraphs
                if not advice:
                    next_elem = section.find_next()
                    count = 0
                    while next_elem and next_elem.name not in ['h2', 'h3'] and count < 3:
                        if next_elem.name == 'p' and next_elem.text.strip():
                            advice.append(next_elem.text.strip())
                            count += 1
                        next_elem = next_elem.find_next()
                
                if advice:
                    break
        
        return advice
    
    def create_fallback_entry(self, condition_name):
        """Create a fallback entry when scraping fails"""
        return {
            "name": condition_name,
            "symptoms": [],
            "advice": [
                f"Consult a healthcare professional for advice on {condition_name}",
                "Always follow the advice of qualified medical practitioners",
                "Do not self-diagnose or self-medicate without professional guidance"
            ],
            "source": "Generic advice (NHS page not available)",
            "url": f"https://www.nhs.uk/search?collection=nhs-meta&q={condition_name.replace(' ', '+')}"
        }
    
    def build_database(self, conditions):
        """Build a database of conditions from NHS website"""
        database = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
        
        # Process each condition
        for condition in tqdm(conditions, desc="Scraping conditions"):
            print(f"\nScraping information for {condition}...")
            result = self.scrape_condition(condition)
            if result:
                database[condition] = result
                
                # Save after each successful scrape to avoid losing data
                with open(self.output_file, 'w') as f:
                    json.dump(database, f, indent=2)
        
        print(f"\nDatabase created with {len(database)} conditions")
        return database


# Main script for execution
if __name__ == "__main__":
    # List of all conditions from the disease dataset
    conditions = [
        "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction",
        "Peptic ulcer diseae", "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
        "Hypertension ", "Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)",
        "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
        "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis",
        "Tuberculosis", "Common Cold", "Pneumonia", "Dimorphic hemmorhoids(piles)",
        "Heart attack", "Varicose veins", "Hypothyroidism", "Hyperthyroidism",
        "Hypoglycemia", "Osteoarthristis", "Arthritis", "Acne", "Urinary tract infection",
        "Psoriasis", "Impetigo", "(vertigo) Paroymsal  Positional Vertigo"
    ]
    
    # Create and run the scraper
    scraper = NHSConditionScraper("medical_advice_database.json")
    scraper.build_database(conditions)