-- comp9311 23T1 Project 1

-- Q1:
create or replace view Q1(subject_code)
as
select subjects.code
from subjects,
     orgunits
where subjects.offeredby = orgunits.id
  and orgunits.longname = 'School of Information Systems, Technology and Management'
  and subjects._equivalent like '%COMP%';

-- Q2:
create or replace view Q2_set(course_id, room)
as
select courses.id, classes.room
from subjects,
     courses,
     semesters,
     classes,
     class_types
where subjects.id = courses.subject
  and courses.semester = semesters.id
  and classes.course = courses.id
  and class_types.id = classes.ctype
  and class_types.unswid = 'LAB'
  and semesters.year = 2011
  and subjects.name ilike '%data%'
;

create or replace view Q2(course_id)
as
select course_id
from (select course_id, count(room) as all_room, count(distinct room) as distint_room
      from Q2_set
      group by course_id) as room_stats
where room_stats.all_room = room_stats.distint_room
;

-- Q3:
create or replace view Q3_courses_with_prof(course_id, semester)
as
select distinct courses.id, courses.semester
from course_staff,
     courses,
     people
where course_staff.course = courses.id
  and course_staff.staff = people.id
  and people.title = 'Prof';

create or replace view Q3(unsw_id, name)
as
select distinct people.unswid, people.name
from students,
     people,
     course_enrolments,
     Q3_courses_with_prof as c
where students.id = people.id
  and course_enrolments.student = students.id
  and course_enrolments.course = c.course_id
  and cast(people.unswid as text) like '3210%'
group by people.unswid, people.name, c.semester
having count(c.course_id) >= 2;

-- Q4:
create or replace view Q4_filtered_students(student_id, program)
as
select distinct pe.student, pe.program
from program_enrolments pe,
     stream_enrolments se,
     streams s,
     stream_types st
where se.partof = pe.id
  and se.stream = s.id
  and s.stype = st.id -- relate stream_type & description
group by pe.student, pe.program
having sum(case when st.description = 'Major' then 1 else 0 end) >= 1
   and sum(case when st.description = 'Research' then 1 else 0 end) >= 1;

create or replace view Q4(unsw_id, program)
as
select distinct ppl.unswid, fs.program
from Q4_filtered_students fs,
     people ppl
where fs.student_id = ppl.id;

-- Q5:
create or replace view Q5_filtered_semesters(semester)
as
select id
from semesters
where year = 2005;

create or replace view Q5(unsw_id, program, course)
as
select ppl.unswid, p.id, c.id
from students st,
     program_enrolments pe,
     programs p,
     course_enrolments ce,
     courses c,
     subjects sb,
     Q5_filtered_semesters fs,
     people ppl
where pe.student = st.id
  and pe.program = p.id
  and pe.semester = fs.semester
  and ce.student = st.id
  and ce.course = c.id
  and c.semester = fs.semester
  and sb.id = c.subject
  and st.id = ppl.id
  and pe.semester = c.semester
  and p.offeredby != sb.offeredby
  and ce.mark = 98;

-- Q6:
create or replace view Q6_filtered_courses_in_2011(course_id, subject)
as
select courses.id, courses.subject
from courses,
     semesters
where courses.semester = semesters.id
  and semesters.year = 2011;

create or replace view Q6_filtered_courses_by_school_in_2011(course_id, orgunit)
as
select fc.course_id, o.id
from Q6_filtered_courses_in_2011 fc,
     subjects sb,
     orgunits o,
     orgunit_types ot
where fc.subject = sb.id
  and sb.offeredby = o.id
  and o.utype = ot.id
  and ot.name = 'School';

create or replace view Q6_filtered_school_buildings_usage(orgunit, building_id)
as
select fc.orgunit, b.id
from Q6_filtered_courses_by_school_in_2011 fc,
     classes cls,
     rooms r,
     buildings b
where fc.course_id = cls.course
  and cls.room = r.id
  and r.building = b.id
group by fc.orgunit, b.id;

create or replace view Q6_school_id_building_count(orgunit, building_count)
as
select fs.orgunit, count(*)
from Q6_filtered_school_buildings_usage fs
group by fs.orgunit;

create or replace view Q6_school_id_building_count_with_name(orgunit, orgunit_name, building_count)
as
select f.orgunit, orgunits.longname, f.building_count
from Q6_school_id_building_count f,
     orgunits
where f.orgunit = orgunits.id;

create or replace view Q6(school_id, building_count)
as
select f.orgunit, f.building_count
from Q6_school_id_building_count_with_name f
where f.building_count > (select subf.building_count
                          from Q6_school_id_building_count_with_name subf
                          where subf.orgunit_name = 'School of Mathematics & Statistics');

-- Q7:
create or replace view Q7_course_facility_list(course, facility)
as
select distinct c.id, rf.facility
from courses c,
     classes cls,
     room_facilities rf
where c.id = cls.course
  and cls.room = rf.room;

create or replace view Q7_course_facility_count(course_id, facility_count)
as
select f.course, count(f.facility) as fc
from Q7_course_facility_list f
group by f.course;

create or replace view Q7_courses_with_most_facilities(course_id)
as
select f.course_id
from Q7_course_facility_count f
where f.facility_count = (select max(subf.facility_count) from Q7_course_facility_count subf);

create or replace view Q7_filtered_course_staff(course_id, staff_id)
as
select distinct cs.course, cs.staff
from Q7_courses_with_most_facilities f,
     course_staff cs,
     affiliations af,
     orgunits org
where f.course_id = cs.course
  and cs.staff = af.staff
  and af.orgunit = org.id
  and org.phone like '9382%';

create or replace view Q7(course_id, staff_name)
as
select f.course_id, ppl.name
from Q7_filtered_course_staff f,
     people ppl
where f.staff_id = ppl.id;


-- Q8:
create or replace view Q8_passed_courses(student_id, semester, uoc)
as
select ce.student, c.semester, sb.uoc
from course_enrolments ce,
     courses c,
     subjects sb
where ce.course = c.id
  and c.subject = sb.id
  and ce.mark >= 60;

create or replace view Q8_semesters_before_2012(semester)
as
select id
from semesters
where year < 2012; -- exclusive (modify if inclusive)

create or replace view Q8_uoc_earned_before_2012_in_program(student_id, program, earned_uoc)
as
select pc.student_id, pe.program, sum(pc.uoc)
from Q8_semesters_before_2012 sem,
     Q8_passed_courses pc,
     program_enrolments pe
where pc.semester = sem.semester -- course passed before 2012
  and pe.semester = pc.semester  --  enrolled in same semester
  and pe.student = pc.student_id
group by pc.student_id, pe.program;

create or replace view Q8_degree_check_list(student_id, program, earned_uoc, expected_uoc, degree_type)
as
select f.student_id, f.program, f.earned_uoc, p.uoc, dt.name
from Q8_uoc_earned_before_2012_in_program f,
     programs p,
     program_degrees pd,
     degree_types dt
where f.program = p.id
  and pd.program = p.id
  and pd.dtype = dt.id;

create or replace view Q8_graduated_bachelors(student_id)
as
select f.student_id
from Q8_degree_check_list f
where earned_uoc >= expected_uoc
  and f.degree_type like '%Bachelor%';

create or replace view Q8_ungraduated_masters(student_id)
as
select distinct f.student_id
from Q8_degree_check_list f,
     Q8_graduated_bachelors b
where f.student_id = b.student_id
  and f.earned_uoc < f.expected_uoc
  and f.degree_type like '%Master%';

create or replace view Q8(unsw_id, name)
as
select ppl.unswid, ppl.name
from Q8_ungraduated_masters f,
     people ppl
where f.student_id = ppl.id;
;

-- Q9:
create or replace function
    Q9(unswid integer) returns setof text
as
$$
DECLARE
    row1          record;
    row2          record;
    student_id    integer;
    semester_date date;
BEGIN
    select ppl.id into student_id from people ppl where ppl.unswid = $1;
    FOR row1 IN
        select c.id as course_code, c.semester as course_semester
        from course_enrolments ce,
             courses c,
             subjects sb
        where ce.student = student_id -- enrolled by given student
          and ce.course = c.id        -- pop course
          and c.subject = sb.id       -- pop subject
          and sb._prereq like '%' || substr(sb.code, 1, 4) || '%' -- courses with same-prefix prereq
        LOOP
            FOR row2 IN
                select distinct ppl.unswid as staff_unswid, s.employed as employed
                from course_staff cs,
                     staff_roles sr,
                     people ppl,
                     staff s
                where cs.course = row1.course_code
                  and cs.role = sr.id
                  and sr.name like '%Tutor%'
                  and cs.staff = ppl.id
                  and s.id = cs.staff
                LOOP
                    select s.starting
                    into semester_date
                    from semesters s
                    where s.id = row1.course_semester; -- pick starting date
                    return next row1.course_code || ' ' || row2.staff_unswid || ' ' || (semester_date - row2.employed);
                END LOOP;
        END LOOP;
END;
$$
    language plpgsql;

-- Q10
create or replace function
    Q10(year courseyeartype, term character(2), orgunit_id integer) returns setof text
as
$$
DECLARE
    row1 record;
    sem  integer; -- semester of given term in given year
BEGIN
    FOR sem IN
        select s.id from semesters s where s.year = $1 and s.term = $2
        LOOP
            FOR row1 IN -- students who enrolled in the program provided by the given orgunit in the given term
                select ppl.unswid as student_unswid, avg(ce.mark) as avg_mark
                from program_enrolments pe,
                     programs p,
                     course_enrolments ce,
                     courses c,
                     people ppl
                where pe.program = p.id                          -- relate program
                  and p.offeredby = $3                           -- program provided by the given orgunit
                  and pe.semester = sem                          -- program in the given term
                  and ce.student = pe.student                    -- student enrolled courses
                  and ce.course = c.id                           -- relate course
                  and c.semester = sem                           -- course in given semester
                  and ce.grade in ('PC', 'PS', 'CR', 'DN', 'HD') -- course passed
                  and ce.mark is not null                        -- mark not null
                  and ppl.id = pe.student                        -- pop student unswid
                group by ppl.unswid
                having avg(ce.mark) > 85
                LOOP
                    return next row1.student_unswid || ' ' || cast(row1.avg_mark as numeric(4, 2));
                END LOOP;
        END LOOP;
END;
$$ language plpgsql;
