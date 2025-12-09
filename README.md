# winter-climb-predictor

This is a tool to predict whether an ice climb in scotland will be in climbable condition on a given day if you input some weather data. The weather data is taken from the 
Cairngorm summit weather station (https://cairngormweather.eps.hw.ac.uk/) and the climbing data from UKClimbing.com 
(https://www.ukclimbing.com/logbook/crags/cairn_gorm_-_cairn_lochan-74/savage_slit_winter-2335). I selected the route as it is one of the most popular and often in condition 
winter routes in the UK, and is located close to a weather station. 

This was completed at the start of a data analytics module I have taken, as a side project to see if I could apply what I already knew about machine learning. As such I would do 
things differently now having completed the module. I ran into several problems on this project, mostly surrounding the dataset. As the day went on I realised that missing and 
clearly incorrect values were common (months with average wind of 0 mph etc). In the end to complete the project by the end of the day I decided to just use temperature. 

Whilst the model is basic I learned a lot about dealing with challenging real-world datasets, and attempting to apply a model to a real-world problem. 
