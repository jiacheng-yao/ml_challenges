SELECT date_trunc('month', created_date::date) AS monthly, 
	concat(10*floor(age/10), '-', 10*floor(age/10) + 10) AS age_group, 
       sum(amount_usd) AS sum_amount, count(user_id) AS volume          
FROM 
(
  select 
    transactions.created_date, transactions.amount_usd, transactions.user_id, 
    (date_part('year', CURRENT_DATE) - users.birth_year) AS age
  from 
    transactions inner join users on transactions.user_id = users.user_id where transactions.transactions_type = 'CARD_PAYMENT'
) AS t
GROUP BY monthly, age_group

