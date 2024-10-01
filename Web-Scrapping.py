from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import os
import requests
import PyPDF2
from io import BytesIO

# Path where to save the letters
output_folder = 'shareholder_letters_text'
os.makedirs(output_folder, exist_ok=True)

# Start Selenium WebDriver
driver = webdriver.Chrome()

# URL of the Berkshire Hathaway shareholder letters page
url = 'https://www.berkshirehathaway.com/letters/letters.html'
driver.get(url)

# Find all the links to the letters (using anchor tags with the year in href)
links = driver.find_elements(By.XPATH, '//a[contains(@href, ".html") or contains(@href, ".pdf")]')

# Function to save PDFs as text
def extract_text_from_pdf(pdf_content):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Extract and download each letter as plain text
for link in links:
    year = link.text
    letter_url = link.get_attribute('href')
    
    # Check if the letter is a PDF or an HTML page
    if letter_url.endswith('.pdf'):
        # Handle PDF letter
        print(f"Downloading PDF letter for {year}...")
        response = requests.get(letter_url)
        if response.status_code == 200:
            pdf_content = response.content
            letter_text = extract_text_from_pdf(pdf_content)
            # Save the plain text from PDF
            with open(os.path.join(output_folder, f'{year}.txt'), 'w', encoding='utf-8') as f:
                f.write(letter_text)
        else:
            print(f"Failed to download PDF for {year}")
    else:
        # Handle HTML letter
        print(f"Downloading HTML letter for {year}...")
        driver.get(letter_url)
        time.sleep(2)
        
        # Get the letter content
        letter_html = driver.page_source
        
        # Use BeautifulSoup to extract plain text from HTML
        soup = BeautifulSoup(letter_html, 'html.parser')
        letter_text = soup.get_text(separator='\n', strip=True)
        
        # Save the plain text to a file named after the year
        with open(os.path.join(output_folder, f'{year}.txt'), 'w', encoding='utf-8') as f:
            f.write(letter_text)
        
        # Return to the main letters page
        driver.get(url)
        time.sleep(1)

# Close the browser
driver.quit()

print(f'Scraped and saved all letters as text to {output_folder}')
