import numpy as np
# date,             0   Reformat: Inlude only year
# price,            1   Reformat: Perhaps do a log transform
# bedrooms,         2   Reformat: Make it into an integer
# bathrooms,--------3   Reformat: Make it into an integer
# sqft_living,------4   Keep
# sqft_lot,---------5   Keep
# floors,           6   Reformat: Make it into an integer
# waterfront,       7   Keep
# view,             8   Keep
# condition,--------9   Delete//////// Replace with isRenovated
# sqft_above,-------10  Delete////////
# sqft_basement,----11  Keep
# yr_built,         12  Delete, replace with house age
# yr_renovated,     13  Delete, replace with isRenovated
# street,           14  Delete////////
# city,             15  Keep
# statezip,---------16  Keep
# country-----------17  Delete////////
#------------------------------------------------
# Things added
# isRenovated:          3 categories, no renovation, old renovation(5+), new renovation(0-5)
# HouseAge:             YearSold - YearBuilt
# Log(Price):           Transform price relative to base


#Sample
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#date           log_price   bedrooms    bathrooms   sqft_living     sqft_lot    floors      waterfront      view        sqft_basement       house_age       isRenovated     city        statezip
#2014-05-09     12.84       3           2           1340            1384        3           0               0           0                   6               Not Renovated   Seattle     WA 98103



# Open the file to read from
with open("data/raw/USA Housing Dataset.csv", "r") as infile:
    # Open the file to write to
    with open("AndresHouseDataOutput.txt", "w") as outfile:
        # Loop through each line in the input file
        for line in infile:
            if(line[0]=="d"):
                outfile.write("date,log_price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,sqft_basement,house_age,isRenovated,city,statezip\n")
                continue
            #Grabs all the factors of each line for furthur reformating, calculation, and deletion
            date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country = line.split(",")
            
            #Changes date
            date = date[:10]

            bedrooms = (float)(bedrooms)
            bedrooms = (int)(bedrooms)

            #Converts bathrooms from float to int
            bathrooms = (float)(bathrooms)
            bathrooms = (int)(bathrooms)

            #Converts floors from float to int
            floors = (float)(floors)
            floors = (int)(floors)

            #Creates 3 categories based on the year the house was renovated
            yr_renovated = (int)(yr_renovated)
            isRenovated = ""
            if (yr_renovated==0):
                isRenovated = "Not Renovated"
            elif(2014-yr_renovated<=5):
                isRenovated = "Recently Renovated"
            else:
                isRenovated = "Older Renovation"

            #Creates a house age variable
            house_age = 2014-int(yr_built)
            
            #Grabs the log of price to the base of e
            log_price = np.log((float)(price)+1)
            
            #Turns back all the maniupulated variables into a string
            bedrooms = (str)(bedrooms)
            bathrooms = (str)(bathrooms)
            floors = (str)(floors)
            house_age = (str)(house_age)
            log_price = f"{log_price:.2f}"

            #Creates a modified line
            outline = ",".join([date,log_price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,sqft_basement,house_age,isRenovated,city,statezip])
            
            # Write the modified line to the output file
            outfile.write(outline+"\n")
        
        


