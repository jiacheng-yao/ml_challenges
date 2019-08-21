SELECT 
    sum(CASE WHEN merged_pivot_table.Nonstandard < merged_pivot_table.Standard THEN 1 else 0 END) as Number_of_Days_Nonstandard_is_Less,
    sum(CASE WHEN merged_pivot_table.Nonstandard >= merged_pivot_table.Standard THEN 1 else 0 END) as Number_of_Days_Nonstandard_is_More
FROM
   (SELECT * 
FROM crosstab( 
'SELECT cast (transaction_date AS text), plan, count_transactions FROM (
SELECT merged_table.plan, merged_table.transaction_date, 
count(merged_table.count_transactions) as count_transactions
 FROM (
SELECT
   users.user_id, CASE WHEN users.plan = ''STANDARD'' THEN ''STANDARD'' ELSE ''NONSTANDARD'' END as plan, transactions_agg.count_transactions, transactions_agg.transaction_date
FROM
   users
INNER JOIN 
(SELECT transactions.user_id, date(transactions.created_date) as transaction_date, 
	count(transactions.transaction_id) as count_transactions
	FROM  transactions group by transactions.user_id, date(transactions.created_date) 
) transactions_agg 
ON users.user_id = transactions_agg.user_id) merged_table
GROUP BY merged_table.plan, merged_table.transaction_date) merged_final_table order by 1,2'
) 
AS final_result(Transaction_Date TEXT, Standard bigint, Nonstandard bigint)) merged_pivot_table