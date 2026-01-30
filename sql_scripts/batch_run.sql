SELECT Date_trunc('hour', ( Cast(datetime AS TIMESTAMP) - Cast(
                               '5 hours' AS INTERVAL) )
          )          AS fecha_hora,
          'w_lluvia' AS medida,
          NULL       AS value_min,
          rain       AS value_avg,
          CASE
            WHEN (( rain > 0 )) THEN ( 100 )
            ELSE 0
          END        AS value_max
   FROM   weather_hourly
   where datetime between '2026-01-26' and '2026-01-27'; 