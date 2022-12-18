# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:30:32 2022

@author: hor
"""

from pymongo import MongoClient
import pandas as pd
from selenium import webdriver
import re
from datetime import datetime
import datetime
import os
from time import sleep
from bs4 import BeautifulSoup
import json

url = 'https://tasty.co/recipe/dairy-free-fettuccine-alfredo'
# url_list_recipes = 'https://www.allrecipes.com/recipes/631/everyday-cooking/seasonal/fall/'


#Initialize browser for crawling and connection to the Database
driver = webdriver.Firefox()
mongo_client = MongoClient()
db = mongo_client['foodiesc']

list_url_that_turned_out_not_to_be_recipes = []

def extract_recipe_from_tasty_dot_co_from_url(url_recipe, manual_added_values_dict = None):
    print('scraping url {}'.format(url_recipe))
    soup_entire_page = BeautifulSoup(driver.find_element_by_tag_name('html').get_attribute('innerHTML'), "html.parser")
    try:
        element_title = driver.find_elements_by_xpath("//h1[@class='recipe-name extra-bold xs-mb05 md-mb1']")[0]
    except:
        # element_title = driver.find_elements_by_xpath("//h1[@id='article-heading_1-0']")[0]
        list_url_that_turned_out_not_to_be_recipes.append(url_recipe)
        return False
    value_title = element_title.text
    
    # Initialize the result Dict object with must-have data
    dict_recipe = {'title' : value_title}
    dict_recipe['url_recipe'] = url_recipe
    dict_recipe['date_scrape'] = datetime.datetime.now()
    
    dict_recipe['raw_data_json'] = json.loads(soup_entire_page.findAll("script", {'type' : 'application/ld+json'})[0].text)
    
    
    
    
    # Scrape ingredients
    soup_ingredients = BeautifulSoup(driver.find_elements_by_class_name('ingredients-prep')[1].get_attribute('innerHTML'), "html.parser")
    list_elements_ingredients = soup_ingredients.find_all('li', {'class' : 'ingredient xs-mb1 xs-mt0'})
    list_ingredients = []
    for element_single_ingredient_li in list_elements_ingredients:
        ingredient_parts_raw = element_single_ingredient_li.encode_contents().decode().split('<!-- -->')
        ingredient_scraped = []
        for part_of_ingredient in ingredient_parts_raw:
            ingredient_scraped.append(part_of_ingredient.strip())
        list_ingredients.append(ingredient_scraped)
    dict_recipe['ingredients'] = list_ingredients 
    
    
    # Scrape date when the recipe was published
    try:
        dict_recipe['date_publish'] = dict_recipe['raw_data_json']['datePublished']
    except:
        pass
    
    # Scrape date when was updated
    try:
        dict_recipe['date_update_of_publish'] = dict_recipe['raw_data_json']['dateModified']
    except:
        pass
    
    # # Scrape the short text paragraph after the title (subheading)
    try:
        dict_recipe['article_subheading'] = dict_recipe['raw_data_json']['description']
    except:
        pass
    
    # Scrape the recipe instruction
    try:
        dict_recipe['instructions'] = driver.find_elements_by_xpath('//ol[@class="prep-steps list-unstyled xs-text-3"]')[1].text
    except:
        pass
    
    if manual_added_values_dict:
        for key in manual_added_values_dict.keys():
            dict_recipe[key] = manual_added_values_dict[key]
    
    return dict_recipe
    

def insert_recipe_dict_into_a_db_document(db, collection, dict_recipe):
    db.get_collection(collection).insert_one(dict_recipe)

def scrape_and_save_single_url_of_tasty_dot_co(url, manual_added_values_dict = None):
    url_recipe = url
    driver.get(url_recipe)
    driver.implicitly_wait(5)
    # try:
    #     driver.find_element_by_id('onetrust-accept-btn-handler').click()
    #     driver.implicitly_wait(1)
    # except:
    #     pass
    dict_recipe = extract_recipe_from_tasty_dot_co_from_url(url_recipe, manual_added_values_dict = manual_added_values_dict)
    try:
        if dict_recipe:
            insert_recipe_dict_into_a_db_document(db, 'recipes_tasty_co', dict_recipe)
    except Exception as e:
        print(e)



def scrape_and_save_to_mongodb_list_of_recipe_url(list_url, manual_added_values_dict = None):
    for url_to_scrape in list_url:
        scrape_and_save_single_url_of_tasty_dot_co(url_to_scrape, manual_added_values_dict = manual_added_values_dict)
        sleep(7)



def get_url_list_of_recipes_from_page_a_loaded_page(driver):
    list_url_results = []
    soup_maindiv = BeautifulSoup(driver.find_element_by_id('all-recipes').get_attribute("innerHTML"), "html.parser")
    list_link_elements = soup_maindiv.findAll("a", {'class' : 'block group'})
    for a_tag in list_link_elements:
        list_url_results.append(a_tag.get_attribute_list('href')[0])
    return list_url_results
    

def get_url_list_of_recipes_from_page_from_url_in_tasty_dot_co(url_list_recipes, pages_number=1):
    driver.get(url_list_recipes)
    driver.implicitly_wait(5)
    
    try:
        for i in range(pages_number):
            driver.find_element_by_class_name('show-more-button').click()
            sleep(5)
    except Exception as e:
        print(e)
    
    list_url_results = []
    list_elemets_a = driver.find_elements_by_xpath("//div[@class='feed__container']/section/ul/li/a")
    for element_url in list_elemets_a:
        list_url_results.append(element_url.get_attribute("href"))
    return list_url_results

url_list_recipes = 'https://tasty.co/topic/lunch'

list_url = get_url_list_of_recipes_from_page_from_url_in_tasty_dot_co(url_list_recipes, pages_number=500)

scrape_and_save_to_mongodb_list_of_recipe_url(list_url, manual_added_values_dict={'found_in_category' : 'lunch'})































