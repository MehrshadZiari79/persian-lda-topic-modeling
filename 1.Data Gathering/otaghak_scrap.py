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

# Function to terminate any running chromedriver.exe processes
def terminate_existing_drivers():
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'chromedriver.exe':
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"Terminated chromedriver.exe process with PID {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                logging.error(f"Error terminating process {proc.info['pid']}")

# Start the timer
start_time = time.time()

# Configure logging
logging.basicConfig(filename='scraping_errors.log', level=logging.ERROR)

# Configure Chrome WebDriver
chrome_driver_path = r'E:\payan\packages\chromedriver-win64\chromedriver.exe'  # Replace with your Chrome WebDriver path
chrome_options = Options()
chrome_options.add_argument("--headless")  # Remove this option for debugging
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--blink-settings=imagesEnabled=false")
chrome_options.add_argument("--disable-javascript")  # Disable JavaScript
chrome_options.add_argument("--disable-stylesheet")  # Disable CSS

# Base URL pattern for room pages
base_url = "https://www.otaghak.com/room/"

# Define the range of page numbers to scrape
start_page = 2411001
end_page = 2411300

# Function to scrape a single page
def scrape_page(driver, page_num):
    data = []
    url = base_url + str(page_num) + "/"
    driver.get(url)

    # Wait for the comments section to be loaded
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".Comments_commentBody__Tl_lO"))
        )
    except TimeoutException:
        print(f"Timeout loading page {url}")
        return data

    # Extract location
    location = "N/A"
    try:
        location_element = driver.find_element(By.CSS_SELECTOR, "a[href^='/province'] span[itemprop='name']")
        location = location_element.text.strip()
        print(f"Location found: {location}")
    except NoSuchElementException:
        print(f"Location element not found on page {page_num}")

    # Extract room type
    room_type = "N/A"
    try:
        room_type_element = driver.find_element(By.CSS_SELECTOR, ".Typography_h1__JESPs.text-Asphalt")
        room_type = room_type_element.text.strip().split()[0]
        print(f"Room type found: {room_type}")
    except NoSuchElementException:
        print(f"Room type element not found on page {page_num}")

    # Scroll the page to load all comments
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    # Extract comments
    comments_containers = driver.find_elements(By.CSS_SELECTOR, ".Comments_commentBody__Tl_lO")
    for container in comments_containers:
        try:
            comment = container.find_element(By.CSS_SELECTOR, ".Typography_caption3__epLLG.text-Charcoal.text-justify")
            text = comment.text.strip()
            data.append({'Page': page_num, 'Location': location, 'Room Type': room_type, 'Comment': text})
            print(f"Saved comment from page {page_num}")
        except NoSuchElementException:
            print(f"Element not found in the comment container on page {page_num}")
            continue

    return data

# Function to scrape pages using a single WebDriver instance
def scrape_pages(driver, pages):
    all_data = []
    for page_num in pages:
        data = scrape_page(driver, page_num)
        all_data.extend(data)
    return all_data

# Number of workers for concurrent scraping
num_workers = 20
pages_per_worker = (end_page - start_page + 1) // num_workers

# Function to initialize WebDriver and scrape pages in a thread
def worker(pages):
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=chrome_options)
    try:
        return scrape_pages(driver, pages)
    finally:
        driver.quit()

# Divide the pages among workers
page_ranges = [range(start_page + i * pages_per_worker, start_page + (i + 1) * pages_per_worker)
               for i in range(num_workers)]

# Use ThreadPoolExecutor to run scraping concurrently
all_data = []
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = executor.map(worker, page_ranges)

    # Collect and batch write data
    batch_size = 100
    batch_data = []
    for result in results:
        batch_data.extend(result)
        if len(batch_data) >= batch_size:
            df = pd.DataFrame(batch_data)
            df.to_csv('comments_selenium.csv', mode='a', header=not os.path.exists('comments_selenium.csv'), index=False)
            batch_data = []  # Reset batch data

    # Write any remaining data
    if batch_data:
        df = pd.DataFrame(batch_data)
        df.to_csv('comments_selenium.csv', mode='a', header=not os.path.exists('comments_selenium.csv'), index=False)

print(f"Scraping completed and data saved to 'comments_selenium.csv'")

# End the timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")
