with staff as (
    select
        e.locationid,
        e.firstname || ' ' || e.lastname as full_name,
        e.lastname,
        e.performance,
        p.positiontitle,
        to_double(replace(replace(e.salary, 'US$', ''), ',', '')) as salary_amount
    from employee e
    left join position p on p.positionid = e.positionid
),
ranked as (
    select
        locationid,
        full_name,
        row_number() over (partition by locationid order by salary_amount desc, lastname asc) as salary_rank
    from staff
),
counts as (
    select
        locationid,
        count(*) as num_employees,
        sum(case when positiontitle = 'Trainee' then 1 else 0 end) as num_of_trainee,
        sum(case when positiontitle = 'Account Representative' and performance = 'Average' then 1 else 0 end)
            as num_account_representatives_with_average_performance,
        sum(case when performance = 'Good' then 1 else 0 end) as num_employees_have_good_perfomance
    from staff
    group by locationid
)
select
    l.locationid as location_id,
    l.state as location_state,
    l.address as location_address,
    coalesce(c.num_employees, 0) as num_employees,
    r.full_name as highest_paid_employee,
    coalesce(c.num_of_trainee, 0) as num_of_trainee,
    coalesce(c.num_account_representatives_with_average_performance, 0)
        as num_account_representatives_with_average_performance,
    coalesce(c.num_employees_have_good_perfomance, 0) as num_employees_have_good_perfomance
from location l
left join counts c on c.locationid = l.locationid
left join ranked r on r.locationid = l.locationid and r.salary_rank = 1
