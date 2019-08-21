CREATE OR REPLACE FUNCTION _final_median(NUMERIC[])
   RETURNS NUMERIC AS
$$
   SELECT AVG(val)
   FROM (
     SELECT val
     FROM unnest($1) val
     ORDER BY 1
     LIMIT  2 - MOD(array_upper($1, 1), 2)
     OFFSET CEIL(array_upper($1, 1) / 2.0) - 1
   ) sub;
$$
LANGUAGE 'sql' IMMUTABLE;
 
CREATE AGGREGATE median(NUMERIC) (
  SFUNC=array_append,
  STYPE=NUMERIC[],
  FINALFUNC=_final_median,
  INITCOND='{}'
);

SELECT country, avg(signup_count), stddev(signup_count), median(signup_count) FROM 
(SELECT date_trunc('week', created_date::date) AS weekly, country, 
       COUNT(user_id) AS signup_count          
FROM users
GROUP BY weekly, country) AS users_aggregate
GROUP BY country;
