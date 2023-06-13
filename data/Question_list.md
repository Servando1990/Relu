
Dataset has 254175 observations

WIthout duplicates, dataset has 223571 observations

Added to Catalog seems to have anomalus values like 1960

Is list price based on added to catalog date or image date?

What is % and % all. They seemed to have the same values. 

Percetages have negative values.

Difference  between stock item and unit of stock?

Units of measurment have zero values. This is corrupting the data.

country of origin have anomalies like "???" 

Product category seems to be made manually. Anmalies like non, "and" and other values spread across the dataset (29760)

Min max of dates

Factors affecting the price elasticity:

Expectations of discounts: holidays, black friday. We can derive this features but we need the date of sales. 

Visibility of discounts markers of price in the website, price tags

clarity oof discounts: promo codes or coupons

Other feartures. duration of promotion, promo type, basket size, range o product the the promo applies 




Quewstions on missing data:
Example 1
Q: Did you notice any instances where some of the data was not recorded? If so, why was this the case?
A: Yes, sometimes data from our South region isn't recorded because the area has poor internet connection and doesn't always successfully upload data.
Nature of Missing Data: Missing Not at Random (MNAR) - the missingness is related to the values of the unobserved data itself, i.e., it depends on the region.
Next Steps: You may need to handle this missing data with care as it is not random. Some methods of dealing with MNAR data are advanced imputation methods or obtaining the missing data by other means if possible.

Example 2
Q: When you saw missing data, was there a common factor? For example, did you notice that it happens more often on certain days, or with certain types of data?
A: Well, we seem to have more missing data during weekends. Our staff is limited during that time, and they might not be able to collect all the necessary data.
Nature of Missing Data: Missing at Random (MAR) - the probability of missing data is related to some observed data. In this case, the day of the week.
Next Steps: You could apply imputation techniques where you fill in the missing data with substitute values. You could base the substitute values on the data collected during the weekdays.

Example 3
Q: Could you please provide any details or notes you might have about when and why some data points are not filled in?
A: Well, some customers don't provide their income level during the survey, so that field is often left blank.
Nature of Missing Data: Missing Completely at Random (MCAR) - The missingness has no relationship with any other data, observed or missing.
Next Steps: If the missing data is MCAR, then it can be safely omitted from the dataset, or simple imputation methods can be used.
