from run_models import  
  


print('before running this script check that you downloaded the twitter embedding and placed it in the right folder')
print('If you want to run our BEST model please enter yes,  then we will run all the preprocessing , the training and the prediction ')
user_input = input(Please enter your choice )
while (user_input != 'yes' and user_input != 'no')
    print(please enter either 'yes' or 'no')
    user_input = input(Please enter your choice )

if user_input == 'yes' 
    print('Your choice was Yes!')
    print('Running BEST MODEL')
    build_dl_model()
else 
    print('Your choice was No!')
    
   
    
    
print('End of the run script!')