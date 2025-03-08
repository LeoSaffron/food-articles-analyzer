# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:36:38 2022

@author: hor
"""

from pymongo import MongoClient
import pandas as pd
from selenium import webdriver
import re
import datetime
from datetime import datetime
import os
from time import sleep
from bs4 import BeautifulSoup

# url = 'https://www.allrecipes.com/recipe/25095/nanaimo-bars-iii/'
url_list_recipes = 'https://www.allrecipes.com/recipes/631/everyday-cooking/seasonal/fall/'


#Initialize browser for crawling and connection to the Database
driver = webdriver.Firefox()
mongo_client = MongoClient()
db = mongo_client['foodiesc']

list_url_that_turned_out_not_to_be_recipes = []

def extract_recipe_from_allrecipes_com_from_url(url_recipe):
    print('scraping url {}'.format(url_recipe))
    try:
        element_title = driver.find_elements_by_xpath("//h1[@id='article-heading_2-0']")[0]
    except:
        # element_title = driver.find_elements_by_xpath("//h1[@id='article-heading_1-0']")[0]
        list_url_that_turned_out_not_to_be_recipes.append(url_recipe)
        return
    value_title = element_title.text
    
    # Initialize the result Dict object with must-have data
    dict_recipe = {'title' : value_title}
    dict_recipe['url_recipe'] = url_recipe
    dict_recipe['date_scrape'] = datetime.datetime.now()
    
    
    # Scrape ingredients
    element_ingredients = driver.find_elements_by_xpath("//div[@class='comp mntl-structured-ingredients']")[0]
    element_ingredients.get_attribute("innerHTML")
    soup = BeautifulSoup(element_ingredients.get_attribute("innerHTML"), "html.parser")
    list_elements_ingredients = soup.findAll('li', {'class': 'mntl-structured-ingredients__list-item'})
    list_ingredients = []
    for element_single_ingredient_li in list_elements_ingredients:
        length_spans_in_ingredients = len(element_single_ingredient_li.findAll('span'))
        if length_spans_in_ingredients == 3:
            value_measurement_number = element_single_ingredient_li.findAll('span')[0].text
            value_measurement_unit = element_single_ingredient_li.findAll('span')[1].text
            value_ingredient = element_single_ingredient_li.findAll('span')[2].text
        else:
            assert(length_spans_in_ingredients == 3, f"number greater snaps in ingredient incorrect. 3 expected, got: {length_spans_in_ingredients}")
        dict_single_ingredient = {'ingredient_name' : value_ingredient,
           'ingredient_measurement_unit' : value_measurement_unit,
           'ingredient_amount' : value_measurement_number}
        list_ingredients.append(dict_single_ingredient)
        
    dict_recipe['ingredients'] = list_ingredients
    
    
    # Scrape date when the recipe was published, or when was updated in case there is no publish date
    soup_entire_page = BeautifulSoup(driver.find_element_by_tag_name('html').get_attribute('innerHTML'), "html.parser")
    element_metadata = soup_entire_page.find("div", id=re.compile("^article-meta-dynamic"))
    try:
        recipe_date_full_str = element_metadata.find("div", {'class' : "mntl-attribution__item-date"}).text
        article_publish_date_str = recipe_date_full_str.split('Published on ')[1]
        article_publish_date_datetime = datetime.datetime.strptime(article_publish_date_str, "%B %d, %Y")
        
        dict_recipe['date_publish'] = article_publish_date_datetime
    except:
        recipe_date_full_str = element_metadata.find("div", {'class' : "mntl-attribution__item-date"}).text
        article_publish_date_str = recipe_date_full_str.split('Updated on ')[1]
        article_publish_date_datetime = datetime.datetime.strptime(article_publish_date_str, "%B %d, %Y")
        
        dict_recipe['date_update_of_publish'] = article_publish_date_datetime
        
    # Scrape the short text paragraph after the title (subheading)
    recipe_article_subheading = None
    element_article_subheading = soup_entire_page.find("p", id=re.compile("^article-subheading_2-0"))
    if element_article_subheading:
        recipe_article_subheading = element_article_subheading.text.strip()
        dict_recipe['article_subheading'] = recipe_article_subheading
        
    
    # Scrape the recipe instruction
    recipe_content = None
    element_recipe_content = soup_entire_page.find("div", id=re.compile("^recipe__steps-content_1"))
    if element_recipe_content:
        recipe_content = element_recipe_content.text.strip()
        dict_recipe['instructions'] = recipe_content
    
    return dict_recipe




def insert_recipe_dict_into_a_db_document(db, collection, dict_recipe):
    db.get_collection(collection).insert_one(dict_recipe)

def scrape_and_save_single_url(url):
    url_recipe = url
    driver.get(url_recipe)
    driver.implicitly_wait(30)
    dict_recipe = extract_recipe_from_allrecipes_com_from_url(url_recipe)
    try:
        insert_recipe_dict_into_a_db_document(db, 'allrecipes_com_recipes', dict_recipe)
    except Exception as e:
        print(e)

def scrape_and_save_to_mongodb_list_of_recipe_url(list_url):
    for url_to_scrape in list_url:
        scrape_and_save_single_url(url_to_scrape)
        sleep(13)

# scrape_and_save_single_url(url)

def get_url_list_of_recipes_from_page_from_url(url_list_recipes):
    driver.get(url_list_recipes)
    soup_maindiv = BeautifulSoup(driver.find_element_by_id('mntl-taxonomysc-article-list-group_1-0').get_attribute("innerHTML"), "html.parser")
    
    list_url_results = []
    list_link_elements = soup_maindiv.findAll("a", id=re.compile("^mntl-card-list-items_"))
    for a_tag in list_link_elements:
        list_url_results.append(a_tag.get_attribute_list('href')[0])
    return list_url_results

# pd.DataFrame(get_url_list_of_recipes_from_page_from_url(url_list_recipes)).to_excel('recipes_to_scrape.xlsx')

def scrape_all_recipes_from_url(url_list_recipes):
    list_url = get_url_list_of_recipes_from_page_from_url(url_list_recipes)
    scrape_and_save_to_mongodb_list_of_recipe_url(list_url)


























