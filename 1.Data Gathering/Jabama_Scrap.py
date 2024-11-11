import concurrent.futures
import time
import pandas as pd
import os
import logging
import psutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# Function to terminate all running chromedriver.exe processes
def terminate_existing_drivers():
    """Terminate any existing chromedriver processes to avoid conflicts."""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'chromedriver.exe':
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"Terminated chromedriver.exe process with PID {proc.info['pid']}")
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                logging.error(f"Access denied to terminate process {proc.info['pid']}")
            except psutil.TimeoutExpired:
                logging.error(f"Timeout expired while waiting to terminate process {proc.info['pid']}")

# Start the timer
start_time = time.time()

# Configure logging
logging.basicConfig(filename='scraping_errors.log', level=logging.ERROR)

# Configure Chrome WebDriver
chrome_driver_path = r'E:\payan\packages\chromedriver-win64\chromedriver.exe'  # Update with correct path
chrome_options = Options()
chrome_options.add_argument("--headless")  # Use headless mode for faster scraping
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_experimental_option("prefs", {"profile.managed_default_content_settings.javascript": 2})  # Disable JavaScript

# Base URL pattern for room pages on Jabama
base_url = "https://www.jabama.com/stay/apartment-"

# Define the range of pages to scrape
start_page = 340000
end_page = 350000

def scrape_page(driver, page_num):
    """Scrape data from a single page."""
    data = []
    url = base_url + str(page_num)
    driver.get(url)

    try:
        # Wait for the comments section to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.comment-card__content p.comment-card__text"))
        )
    except TimeoutException:
        print(f"Timeout loading page {url}")
        return data

    # Extract location and room type
    location = extract_element(driver, "strong.city-province", "city-province", "N/A")
    room_type = extract_element(driver, "h2.pdp-host-info-content__title", "room type", "N/A")

    # Scroll the page to load all comments
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # Extract comments
    comments_containers = driver.find_elements(By.CSS_SELECTOR, "div.comment-card__content p.comment-card__text")
    print(f"Found {len(comments_containers)} comment containers on page {page_num}")

    for container in comments_containers:
        try:
            text = container.text.strip()
            if text:
                data.append({'Page': page_num, 'Location': location, 'Room Type': room_type, 'Comment': text})
        except Exception as e:
            print(f"Error extracting comment on page {page_num}: {e}")
            continue

    return data

def extract_element(driver, selector, element_type, default_value):
    """Extract the element text from the page or return default value."""
    try:
        element = driver.find_element(By.CSS_SELECTOR, selector)
        return element.text.strip().split('،')[-1].strip() if element_type == "city-province" else element.text.strip()
    except NoSuchElementException:
        print(f"{element_type} not found on page")
        return default_value

def scrape_pages(driver, pages):
    """Scrape a list of pages using a single driver."""
    all_data = []
    for page_num in pages:
        data = scrape_page(driver, page_num)
        all_data.extend(data)
    return all_data

def worker(pages):
    """Worker function to initialize WebDriver and scrape pages."""
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=chrome_options)
    try:
        return scrape_pages(driver, pages)
    finally:
        driver.quit()

# Divide pages among workers
num_workers = 30
pages_per_worker = (end_page - start_page + 1) // num_workers
page_ranges = [range(start_page + i * pages_per_worker, start_page + (i + 1) * pages_per_worker) for i in range(num_workers)]

# Use ThreadPoolExecutor for concurrent scraping
all_data = []
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = executor.map(worker, page_ranges)

    # Batch write data to CSV
    batch_size = 750
    batch_data = []
    for result in results:
        batch_data.extend(result)
        if len(batch_data) >= batch_size:
            df = pd.DataFrame(batch_data)
            df.to_csv('cm_jabama.csv', mode='a', header=not os.path.exists('cm_jabama.csv'), index=False)
            batch_data = []

    if batch_data:
        df = pd.DataFrame(batch_data)
        df.to_csv('cm_jabama.csv', mode='a', header=not os.path.exists('cm_jabama.csv'), index=False)

print(f"Scraping completed and data saved to 'cm_jabama.csv'")

# End the timer and print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")
