select start_time, execution_time
from calendar_executions
where executed = 1;
/*
┌────────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│  column_name   │ column_type │  null   │   key   │ default │  extra  │
│    varchar     │   varchar   │ varchar │ varchar │ varchar │ varchar │
├────────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ execution_id   │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ start_time     │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ execution_time │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ code_on        │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ code_off       │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ executed       │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
*/