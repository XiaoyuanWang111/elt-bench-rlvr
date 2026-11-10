with employee_salary as (
    select
        positionid,
        gender,
        performance,
        to_double(replace(replace(salary, 'US$', ''), ',', '')) as salary_amount
    from employee
)
select
    p.positionid as position_id,
    p.positiontitle as position_title,
    p.educationrequired as education_required,
    coalesce(sum(case when e.performance = 'Poor' then 1 else 0 end), 0) as num_poor_performing_employees,
    coalesce(sum(case when e.gender = 'F' then 1 else 0 end), 0) as num_female_employees,
    100.0 * sum(case when e.salary_amount > 20000 then 1 else 0 end) / nullif(count(e.positionid), 0)
        as per_salary_over_20000,
    avg(e.salary_amount) as avg_salary,
    case
        when to_double(replace(replace(p.maxsalary, 'US$', ''), ',', '')) <= 70000 then 1
        else 0
    end as has_maximum_salary_no_more_than_70000
from position p
left join employee_salary e on e.positionid = p.positionid
group by p.positionid, p.positiontitle, p.educationrequired, p.maxsalary
