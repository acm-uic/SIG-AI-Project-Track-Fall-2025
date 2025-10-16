# date,             0   Reformat: Inlude only year
# price,            1   Reformat: Perhaps do a log transform
# bedrooms,         2   Reformat: Make it into an integer
# bathrooms,--------3   Reformat: Make it into an integer
# sqft_living,------4   Keep
# sqft_lot,---------5   Keep
# floors,           6   Reformat: Make it into an integer
# waterfront,       7   Keep
# view,             8   Keep
# condition,--------9   Keep
# sqft_above,-------10  Delete////////
# sqft_basement,----11  Keep
# yr_built,         12  Keep
# yr_renovated,     13  Delete/Reformat heavily: I might replace it with a year renovated catagory
# street,           14  Delete////////
# city,             15  Keep
# statezip,---------16  Keep
# country-----------17  Delete////////
#------------------------------------------------
# Things to add
# isRenovated:          A boolean that is true depending on how long ago the house was renovated before being sold
# HouseAge:             YearSold-YearBuilt
# Log(Price):           Transform price relative to base
#

#date                   price       bedrooms    bathrooms   sqft_living     sqft_lot    floors  waterfront  view    condition       sqft_above      sqft_basement   yr_built    yr_renovated    street                      city        statezip    country
#2014-05-09 00:00:00    376000.0    3.0         2.0         1340            1384        3.0     0           0       3               1340            0               2008        0               9245-9249 Fremont Ave N     Seattle     WA 98103    USA



# Open the file you want to read from
with open("data/raw/USA Housing Dataset.csv", "r") as infile:
    # Open the file you want to write to
    with open("output.txt", "w") as outfile:
        # Loop through each line in the input file
        for line in infile:
            #rowOfData = line.split(",")

            
            # Write the modified line to the output file
            outfile.write(line)


