-- Q1_scope
SELECT COUNT(*) sessions,COUNT(DISTINCT ChargerID) chargers,SUM(Demand) total_kWh,AVG(Demand) mean_kWh FROM sessions;

-- Q2_hour
SELECT connection_hour,COUNT(*) sessions,SUM(Demand) total_kWh,AVG(Demand) mean_kWh FROM sessions GROUP BY connection_hour ORDER BY connection_hour;

-- Q3_weekday
SELECT dow,COUNT(*) sessions,SUM(Demand) total_kWh,AVG(Demand) mean_kWh FROM sessions GROUP BY dow ORDER BY dow;

-- Q4_month
SELECT year_month,COUNT(*) sessions,SUM(Demand) total_kWh FROM sessions GROUP BY year_month ORDER BY year_month;

-- Q5_charger
WITH a AS (SELECT ChargerID,COUNT(*) sessions,ROUND(SUM(Demand),6) total_kWh FROM sessions GROUP BY ChargerID)
 SELECT *,RANK() OVER(ORDER BY total_kWh DESC) energy_rank FROM a ORDER BY total_kWh DESC,ChargerID;

-- Q6_identity
SELECT known_user,COUNT(*) sessions,AVG(Demand) mean_kWh FROM sessions GROUP BY known_user ORDER BY known_user;

-- Q7_hotspot
SELECT dow,connection_hour,COUNT(*) sessions,SUM(Demand) total_kWh FROM sessions GROUP BY dow,connection_hour ORDER BY dow,connection_hour;

-- Q8_location
SELECT c.Location,COUNT(*) sessions,SUM(s.Demand) total_kWh,AVG(s.Demand) mean_kWh
 FROM sessions s JOIN chargers c ON s.ChargerID=c.ChargerID GROUP BY c.Location ORDER BY c.Location;

-- Q9_share
WITH a AS(SELECT ChargerID,SUM(Demand) total_kWh FROM sessions GROUP BY ChargerID)
 SELECT *,100.0*total_kWh/SUM(total_kWh) OVER() pct_energy FROM a ORDER BY ChargerID;