insert into all_executions
(select 
    'calendar' as _source,
    CAST(ce.start_time AS TIMESTAMP) as start_time, 
    ce.execution_time as runtime
from calendar_executions ce 
left join all_executions ae
    on ae._source = 'calendar'
    and ce.start_time = ae.start_time
    and ce.execution_time = ae.runtime
where ce.executed = 1 and ae.start_time is null)
union all
(select 
    'schedule' as _source,
    cast(sd.last_execution as timestamp) as start_time,
    sc.execution_time as runtime
from schedule_data as sd 
inner join schedule as sc 
    on sd.schedule_id = sc.schedule_id
left join all_executions ae
    on ae._source = 'schedule'
    and sd.last_execution = ae.start_time
    and sc.execution_time = ae.runtime
where ae.start_time is null);