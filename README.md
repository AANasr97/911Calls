# 911 Calls Project

## Overview

### Understanding the Background

* Montgomery County

Montgomery County, locally also referred to as Montco, is a county located in the Commonwealth of Pennsylvania. As of the 2010 census, the population was 799,874, making it the third-most populous county in Pennsylvania, after Philadelphia and Allegheny Counties. The county seat is Norristown. Montgomery County is very diverse, ranging from farms and open land in Upper Hanover to densely populated rowhouse streets in Cheltenham.

* 911 Calls

Created by Congress in 2004 as the 911 Implementation and Coordination Office (ICO), the National 911 Program is housed within the National Highway Traffic Safety Administration at the U.S. Department of Transportation and is a joint program with the National Telecommunication and Information Administration in the Department of Commerce.

### Goal:

* Locations from which 911 calls are most frequent
* Time daily, month, weekly patterns of 911 calls
* Major Causes of 911 calls


**This analysis will help to deploy more agents in specific location and save/help people at right time**

---
----

### The Data

`Acknowledgements`: Data provided by  <a href='http://www.pieriandata.com'>montcoalert.org</a>

we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:

Column | Definition
--- | -----------
lat | Latitude
lng | Longitude
desc | Description of the Emergency Call
zip | Zipcode
title | Title of Emergency
timeStamp | YYYY-MM-DD HH:MM:SS
twp | Township
addr | Address
e | Dummy variable (always 1)

