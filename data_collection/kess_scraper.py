from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
from dotenv import load_dotenv

class KESSScraper:
    def __init__(self):
        load_dotenv()
        self.base_url = "https://kess.kedi.re.kr/index"
        self.driver = None
        self.wait = None
        
    def setup_driver(self):
        """Set up Chrome WebDriver with appropriate options"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def login(self):
        """Login to KESS website"""
        self.driver.get(self.base_url)
        
        # Wait for login elements
        login_button = self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='login']"))
        )
        login_button.click()
        
        # Enter credentials
        username = os.getenv('KESS_USERNAME')
        password = os.getenv('KESS_PASSWORD')
        
        if not username or not password:
            raise ValueError("KESS credentials not found in environment variables")
            
        self.driver.find_element(By.NAME, "username").send_keys(username)
        self.driver.find_element(By.NAME, "password").send_keys(password)
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        
    def navigate_to_statistics(self):
        """Navigate to the school statistics page"""
        # Wait for navigation menu
        statistics_link = self.wait.until(
            EC.element_to_be_clickable((By.LINK_TEXT, "통계자료"))
        )
        statistics_link.click()
        
        # Navigate to school statistics
        school_stats_link = self.wait.until(
            EC.element_to_be_clickable((By.LINK_TEXT, "학교통계"))
        )
        school_stats_link.click()
        
    def select_parameters(self, year):
        """Select parameters for data extraction"""
        # Select year
        year_select = self.wait.until(
            EC.element_to_be_clickable((By.NAME, "year"))
        )
        year_select.click()
        self.driver.find_element(By.XPATH, f"//option[text()='{year}']").click()
        
        # Select school type (초등학교)
        school_type = self.wait.until(
            EC.element_to_be_clickable((By.NAME, "schoolType"))
        )
        school_type.click()
        self.driver.find_element(By.XPATH, "//option[text()='초등학교']").click()
        
        # Select region (전국)
        region = self.wait.until(
            EC.element_to_be_clickable((By.NAME, "region"))
        )
        region.click()
        self.driver.find_element(By.XPATH, "//option[text()='전국']").click()
        
    def extract_data(self):
        """Extract school and student data"""
        # Click search button
        search_button = self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
        )
        search_button.click()
        
        # Wait for results table
        table = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.result-table"))
        )
        
        # Extract data from table
        data = []
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        for row in rows[1:]:  # Skip header row
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 3:  # Ensure we have enough columns
                school_name = cols[0].text
                class_count = cols[1].text
                student_count = cols[2].text
                data.append({
                    'school_name': school_name,
                    'class_count': class_count,
                    'student_count': student_count
                })
        
        return pd.DataFrame(data)
        
    def scrape_data(self, year):
        """Main method to scrape data for a specific year"""
        try:
            self.setup_driver()
            self.login()
            self.navigate_to_statistics()
            self.select_parameters(year)
            df = self.extract_data()
            return df
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
                
    def save_data(self, df, year):
        """Save scraped data to CSV file"""
        if df is not None:
            output_dir = "data/raw"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"kess_school_data_{year}.csv")
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Data saved to {output_file}")
            
if __name__ == "__main__":
    scraper = KESSScraper()
    # Scrape data for the last 5 years
    for year in range(2020, 2025):
        print(f"Scraping data for year {year}...")
        df = scraper.scrape_data(year)
        if df is not None:
            scraper.save_data(df, year) 