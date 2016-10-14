# HackerRank: Predict Email Opens
<b>a Simple XGBoost Example</b>
 
<b>Task</b>

HackerRank provides metadata for emails sent to HackerRank users over a certain period of time. This metadata contains specific information about:

- The user the email was sent to.
- The email that was sent.
- The user's reaction to the email, including (among other things) whether or not the user opened the email.

Given the metadata for additional future emails, you must predict whether or not each user will open an email.

<b>Dataset</b>

training_dataset.csv.zip: This file contains the details for various emails sent. If an email was opened, then the value of the opened attribute is ; otherwise, its value is .

test_dataset.csv.zip: This file contains the test dataset. All fields relevant to the user's reaction to the email are missing in this dataset. You must predict the value of the opened attribute for each email.

attributes.pdf: This file contains definitions for all the attributes given in the dataset.
