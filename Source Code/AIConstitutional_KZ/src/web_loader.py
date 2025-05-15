from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Configure browser options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run without opening a browser window
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Initialize the Chrome browser with specified options
browser = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), 
    options=chrome_options
)

# Target URL of the Constitution
target_url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
browser.get(target_url)

# Allow time for the page to fully load
time.sleep(5)

# Extract the visible page text
page_text = browser.find_element("tag name", "body").text

# Write the content to a local file
with open("constitution.txt", "w", encoding="utf-8") as file:
    file.write(page_text)

# Close the browser
browser.quit()
print("✅ Constitution successfully saved to constitution.txt")