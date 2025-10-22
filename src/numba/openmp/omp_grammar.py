openmp_grammar = r"""
    openmp_statement: openmp_construct
                    | openmp_directive
    openmp_directive: barrier_directive
                    | taskwait_directive
                    | taskyield_directive
                    | flush_directive
    barrier_directive: BARRIER
    taskwait_directive: TASKWAIT
    taskyield_directive: TASKYIELD
    BARRIER: "barrier"
    TASKWAIT: "taskwait"
    TASKYIELD: "taskyield"
    taskgroup_directive: TASKGROUP
    taskgroup_construct: taskgroup_directive
    TASKGROUP: "taskgroup"
    openmp_construct: parallel_construct
                    | parallel_for_construct
                    | for_construct
                    | single_construct
                    | task_construct
                    | teams_construct
                    | teams_distribute_construct
                    | teams_distribute_simd_construct
                    | teams_distribute_parallel_for_construct
                    | teams_distribute_parallel_for_simd_construct
                    | loop_construct
                    | teams_loop_construct
                    | target_construct
                    | target_teams_construct
                    | target_teams_distribute_construct
                    | target_teams_distribute_simd_construct
                    | target_teams_distribute_parallel_for_simd_construct
                    | target_teams_distribute_parallel_for_construct
                    | target_loop_construct
                    | target_teams_loop_construct
                    | target_enter_data_construct
                    | target_exit_data_construct
                    | distribute_construct
                    | distribute_simd_construct
                    | distribute_parallel_for_construct
                    | distribute_parallel_for_simd_construct
                    | critical_construct
                    | atomic_construct
                    | sections_construct
                    | section_construct
                    | simd_construct
                    | for_simd_construct
                    | parallel_for_simd_construct
                    | target_data_construct
                    | target_update_construct
                    | parallel_sections_construct
                    | master_construct
                    | ordered_construct
    for_simd_construct: for_simd_directive
    for_simd_directive: FOR SIMD [for_simd_clause*]
    for_simd_clause: for_clause
                   | simd_clause
    parallel_for_simd_construct: parallel_for_simd_directive
    parallel_for_simd_directive: PARALLEL FOR SIMD [parallel_for_simd_clause*]
    parallel_for_simd_clause: parallel_for_clause
                            | simd_clause
    distribute_construct: distribute_directive
    distribute_simd_construct: distribute_simd_directive
    distribute_directive: DISTRIBUTE [distribute_clause*]
    distribute_simd_directive: DISTRIBUTE SIMD [distribute_simd_clause*]
    distribute_clause: private_clause
                     | firstprivate_clause
              //     | lastprivate_distribute_clause
                     | collapse_clause
                     | dist_schedule_clause
                     | allocate_clause
    distribute_simd_clause: private_clause
                          | firstprivate_clause
                   //     | lastprivate_distribute_clause
                          | collapse_clause
                          | dist_schedule_clause
                          | allocate_clause
                          | if_clause
                   //     | safelen_clause
                   //     | simdlen_clause
                          | linear_clause
                          | aligned_clause
                   //     | nontemporal_clause
                          | reduction_clause
                   //     | order_clause

    teams_distribute_clause: num_teams_clause
                           | thread_limit_clause
                           | data_default_clause
                           | private_clause
                           | firstprivate_clause
                           | data_sharing_clause
                           | reduction_clause
                           | allocate_clause
                    //     | lastprivate_distribute_clause
                           | collapse_clause
                           | dist_schedule_clause
                           | ompx_attribute

    teams_distribute_simd_clause: num_teams_clause
                                | thread_limit_clause
                                | data_default_clause
                                | private_clause
                                | firstprivate_clause
                                | data_sharing_clause
                                | reduction_clause
                                | allocate_clause
                         //     | lastprivate_distribute_clause
                                | collapse_clause
                                | dist_schedule_clause
                                | if_clause
                         //     | safelen_clause
                         //     | simdlen_clause
                                | linear_clause
                                | aligned_clause
                         //     | nontemporal_clause
                         //     | order_clause
                                | ompx_attribute

    distribute_parallel_for_construct: distribute_parallel_for_directive
    distribute_parallel_for_directive: DISTRIBUTE PARALLEL FOR [distribute_parallel_for_clause*]
    distribute_parallel_for_clause: if_clause
                                  | num_threads_clause
                                  | data_default_clause
                                  | private_clause
                                  | firstprivate_clause
                                  | data_sharing_clause
                                  | reduction_clause
                                  | copyin_clause
                           //     | proc_bind_clause
                                  | allocate_clause
                                  | lastprivate_clause
                                  | linear_clause
                                  | schedule_clause
                                  | collapse_clause
                                  | ORDERED
                                  | NOWAIT
                           //     | order_clause
                                  | dist_schedule_clause

    distribute_parallel_for_simd_construct: distribute_parallel_for_simd_directive
    distribute_parallel_for_simd_directive: DISTRIBUTE PARALLEL FOR SIMD [distribute_parallel_for_simd_clause*]
    distribute_parallel_for_simd_clause: if_clause
                                  | num_threads_clause
                                  | data_default_clause
                                  | private_clause
                                  | firstprivate_clause
                                  | data_sharing_clause
                                  | reduction_clause
                                  | copyin_clause
                           //     | proc_bind_clause
                                  | allocate_clause
                                  | lastprivate_clause
                                  | linear_clause
                                  | schedule_clause
                                  | collapse_clause
                                  | ORDERED
                                  | NOWAIT
                           //     | order_clause
                                  | dist_schedule_clause
                           //     | safelen_clause
                           //     | simdlen_clause
                                  | aligned_clause
                           //     | nontemporal_clause

    target_data_construct: target_data_directive
    target_data_directive: TARGET DATA [target_data_clause*]
    DATA: "data"
    ENTER: "enter"
    EXIT: "exit"
    target_enter_data_construct: target_enter_data_directive
    target_enter_data_directive: TARGET ENTER DATA [target_enter_data_clause*]
    target_exit_data_construct: target_exit_data_directive
    target_exit_data_directive: TARGET EXIT DATA [target_exit_data_clause*]
    target_data_clause: device_clause
                      | map_clause
                      | if_clause
                      | NOWAIT
                      | depend_with_modifier_clause
    target_enter_data_clause: device_clause
                            | map_enter_clause
                            | if_clause
                            | NOWAIT
                            | depend_with_modifier_clause
    target_exit_data_clause: device_clause
                           | map_exit_clause
                           | if_clause
                           | NOWAIT
                           | depend_with_modifier_clause
    device_clause: "device" "(" const_num_or_var ")"
    map_clause: "map" "(" [map_type ":"] var_list ")"
    map_type: ALLOC | TO | FROM | TOFROM
    map_enter_clause: "map" "(" map_enter_type ":" var_list ")"
    map_enter_type: ALLOC | TO
    map_exit_clause: "map" "(" map_exit_type ":" var_list ")"
    map_exit_type: FROM | RELEASE | DELETE
    update_motion_type: TO | FROM
    TO: "to"
    FROM: "from"
    ALLOC: "alloc"
    TOFROM: "tofrom"
    RELEASE: "release"
    DELETE: "delete"
    parallel_sections_construct: parallel_sections_directive
    parallel_sections_directive: PARALLEL SECTIONS [parallel_sections_clause*]
    parallel_sections_clause: unique_parallel_clause
                            | data_default_clause
                            | private_clause
                            | firstprivate_clause
                            | lastprivate_clause
                            | data_sharing_clause
                            | reduction_clause
    sections_construct: sections_directive
    sections_directive: SECTIONS [sections_clause*]
    SECTIONS: "sections"
    sections_clause: private_clause
                   | firstprivate_clause
                   | lastprivate_clause
                   | reduction_clause
                   | NOWAIT
    section_construct: section_directive
    section_directive: SECTION
    SECTION: "section"
    atomic_construct: atomic_directive
    atomic_directive: ATOMIC [atomic_clause] [seq_cst_clause]
    ATOMIC: "atomic"
    atomic_clause: READ
                 | WRITE
                 | UPDATE
                 | CAPTURE
    READ: "read"
    WRITE: "write"
    UPDATE: "update"
    CAPTURE: "capture"
    seq_cst_clause: "seq_cst"
    critical_construct: critical_directive
    critical_directive: CRITICAL
    CRITICAL: "critical"
    teams_construct: teams_directive
    teams_directive: TEAMS [teams_clause*]
    teams_distribute_directive: TEAMS DISTRIBUTE [teams_distribute_clause*]
    teams_distribute_simd_directive: TEAMS DISTRIBUTE SIMD [teams_distribute_simd_clause*]
    target_construct: target_directive
    target_teams_distribute_parallel_for_simd_construct: target_teams_distribute_parallel_for_simd_directive
    target_teams_distribute_parallel_for_construct: target_teams_distribute_parallel_for_directive
    teams_distribute_parallel_for_construct: teams_distribute_parallel_for_directive
    teams_distribute_parallel_for_simd_construct: teams_distribute_parallel_for_simd_directive
    loop_construct: loop_directive
    teams_loop_construct: teams_loop_directive
    target_loop_construct: target_loop_directive
    target_teams_loop_construct: target_teams_loop_directive
    target_teams_construct: target_teams_directive
    target_teams_distribute_construct: target_teams_distribute_directive
    target_teams_distribute_simd_construct: target_teams_distribute_simd_directive
    teams_distribute_construct: teams_distribute_directive
    teams_distribute_simd_construct: teams_distribute_simd_directive
    target_directive: TARGET [target_clause*]
    HAS_DEVICE_ADDR: "has_device_addr"
    has_device_addr_clause: HAS_DEVICE_ADDR "(" var_list ")"
    target_clause: if_clause
                 | device_clause
                 | thread_limit_clause
                 | private_clause
                 | firstprivate_clause
          //     | in_reduction_clause
                 | map_clause
                 | is_device_ptr_clause
                 | has_device_addr_clause
          //     | defaultmap_clause
                 | NOWAIT
                 | allocate_clause
                 | depend_with_modifier_clause
          //     | uses_allocators_clause
                 | ompx_attribute
    teams_clause: num_teams_clause
                | thread_limit_clause
                | data_default_clause
                | private_clause
                | firstprivate_clause
                | data_sharing_clause
                | reduction_clause
                | allocate_clause
    num_teams_clause: NUM_TEAMS "(" const_num_or_var ")"
    NUM_TEAMS: "num_teams"
    thread_limit_clause: THREAD_LIMIT "(" const_num_or_var ")"
    THREAD_LIMIT: "thread_limit"

    dist_schedule_expr: DIST_SCHEDULE "(" STATIC ")"
    dist_schedule_no_expr: DIST_SCHEDULE "(" STATIC "," const_num_or_var ")"
    dist_schedule_clause: dist_schedule_expr
                        | dist_schedule_no_expr
    DIST_SCHEDULE: "dist_schedule"

    target_teams_distribute_parallel_for_simd_directive: TARGET TEAMS DISTRIBUTE PARALLEL FOR SIMD [target_teams_distribute_parallel_for_simd_clause*]
    target_teams_distribute_parallel_for_simd_clause: if_clause
                                                    | device_clause
                                                    | private_clause
                                                    | firstprivate_clause
                                             //     | in_reduction_clause
                                                    | map_clause
                                                    | is_device_ptr_clause
                                             //     | defaultmap_clause
                                                    | NOWAIT
                                                    | allocate_clause
                                                    | depend_with_modifier_clause
                                             //     | uses_allocators_clause
                                                    | num_teams_clause
                                                    | thread_limit_clause
                                                    | data_default_clause
                                                    | data_sharing_clause
                                                    | reduction_clause
                                                    | num_threads_clause
                                                    | copyin_clause
                                             //     | proc_bind_clause
                                                    | lastprivate_clause
                                                    | linear_clause
                                                    | schedule_clause
                                                    | collapse_clause
                                                    | ORDERED
                                             //     | order_clause
                                                    | dist_schedule_clause
                                             //     | safelen_clause
                                             //     | simdlen_clause
                                                    | aligned_clause
                                             //     | nontemporal_clause
                                                    | ompx_attribute

    teams_distribute_parallel_for_simd_directive: TEAMS DISTRIBUTE PARALLEL FOR SIMD [teams_distribute_parallel_for_simd_clause*]
    teams_distribute_parallel_for_simd_clause: num_teams_clause
                                             | thread_limit_clause
                                      //     | default_clause
                                             | private_clause
                                             | firstprivate_clause
                                             | data_sharing_clause
                                             | reduction_clause
                                             | if_clause
                                             | num_threads_clause
                                             | copyin_clause
                                      //     | proc_bind_clause
                                             | lastprivate_clause
                                             | linear_clause
                                             | schedule_clause
                                             | collapse_clause
                                             | ORDERED
                                             | NOWAIT
                                      //     | order_clause
                                             | dist_schedule_clause
                                      //     | safelen_clause
                                      //     | simdlen_clause
                                             | aligned_clause
                                      //     | nontemporal_clause
                                      //     | in_reduction_clause
                                             | map_clause
                                             | is_device_ptr_clause
                                      //     | defaultmap_clause
                                             | allocate_clause
                                             | depend_with_modifier_clause
                                      //     | uses_allocators_clause
                                             | data_default_clause
                                             | ompx_attribute

    target_teams_distribute_parallel_for_directive: TARGET TEAMS DISTRIBUTE PARALLEL FOR [target_teams_distribute_parallel_for_clause*]
    target_teams_distribute_parallel_for_clause: if_clause
                                               | device_clause
                                               | private_clause
                                               | firstprivate_clause
                                        //     | in_reduction_clause
                                               | map_clause
                                               | is_device_ptr_clause
                                        //     | defaultmap_clause
                                               | NOWAIT
                                               | allocate_clause
                                               | depend_with_modifier_clause
                                        //     | uses_allocators_clause
                                               | num_teams_clause
                                               | thread_limit_clause
                                               | data_default_clause
                                               | data_sharing_clause
                                               | reduction_clause
                                               | num_threads_clause
                                               | copyin_clause
                                        //     | proc_bind_clause
                                               | lastprivate_clause
                                               | linear_clause
                                               | schedule_clause
                                               | collapse_clause
                                               | ORDERED
                                        //     | order_clause
                                               | dist_schedule_clause
                                               | ompx_attribute

    teams_distribute_parallel_for_directive: TEAMS DISTRIBUTE PARALLEL FOR [teams_distribute_parallel_for_clause*]
    teams_distribute_parallel_for_clause: num_teams_clause
                                        | thread_limit_clause
                                        | data_default_clause
                                        | private_clause
                                        | firstprivate_clause
                                        | data_sharing_clause
                                        | reduction_clause
                                        | allocate_clause
                                        | if_clause
                                        | num_threads_clause
                                        | copyin_clause
                                 //     | proc_bind_clause
                                        | lastprivate_clause
                                        | linear_clause
                                        | schedule_clause
                                        | collapse_clause
                                        | ORDERED
                                        | NOWAIT
                                 //     | order_clause
                                        | dist_schedule_clause
                                        | ompx_attribute

    LOOP: "loop"

    ompx_attribute: OMPX_ATTRIBUTE "(" PYTHON_NAME "(" number_list ")" ")"
    OMPX_ATTRIBUTE: "ompx_attribute"
    loop_directive: LOOP [teams_distribute_parallel_for_clause*]
    teams_loop_directive: TEAMS LOOP [teams_distribute_parallel_for_clause*]
    target_loop_directive: TARGET LOOP [target_teams_distribute_parallel_for_clause*]
    target_teams_loop_directive: TARGET TEAMS LOOP [target_teams_distribute_parallel_for_clause*]

    target_teams_directive: TARGET TEAMS [target_teams_clause*]
    target_teams_clause: if_clause
                       | device_clause
                       | private_clause
                       | firstprivate_clause
                //     | in_reduction_clause
                       | map_clause
                       | is_device_ptr_clause
                //     | defaultmap_clause
                       | NOWAIT
                       | allocate_clause
                       | depend_with_modifier_clause
                //     | uses_allocators_clause
                       | num_teams_clause
                       | thread_limit_clause
                       | data_default_clause
                       | data_sharing_clause
                //     | reduction_default_only_clause
                       | reduction_clause
                       | ompx_attribute

    target_teams_distribute_simd_directive: TARGET TEAMS DISTRIBUTE SIMD [target_teams_distribute_simd_clause*]
    target_teams_distribute_simd_clause: if_clause
                                       | device_clause
                                       | private_clause
                                       | firstprivate_clause
                                //     | in_reduction_clause
                                       | map_clause
                                       | is_device_ptr_clause
                                //     | defaultmap_clause
                                       | NOWAIT
                                       | allocate_clause
                                       | depend_with_modifier_clause
                                //     | uses_allocators_clause
                                       | num_teams_clause
                                       | thread_limit_clause
                                       | data_default_clause
                                       | data_sharing_clause
                                       | reduction_clause
                                //     | reduction_default_only_clause
                                       | lastprivate_clause
                                       | collapse_clause
                                       | dist_schedule_clause
                                //     | safelen_clause
                                //     | simdlen_clause
                                       | linear_clause
                                       | aligned_clause
                                //     | nontemporal_clause
                                //     | order_clause
                                       | ompx_attribute

    target_teams_distribute_directive: TARGET TEAMS DISTRIBUTE [target_teams_distribute_clause*]
    target_teams_distribute_clause: if_clause
                                  | device_clause
                                  | private_clause
                                  | firstprivate_clause
                           //     | in_reduction_clause
                                  | map_clause
                                  | is_device_ptr_clause
                           //     | defaultmap_clause
                                  | NOWAIT
                                  | allocate_clause
                                  | depend_with_modifier_clause
                           //     | uses_allocators_clause
                                  | num_teams_clause
                                  | thread_limit_clause
                                  | data_default_clause
                                  | data_sharing_clause
                           //     | reduction_default_only_clause
                                  | reduction_clause
                                  | lastprivate_clause
                                  | collapse_clause
                                  | dist_schedule_clause
                                  | ompx_attribute

    IS_DEVICE_PTR: "is_device_ptr"
    is_device_ptr_clause: IS_DEVICE_PTR "(" var_list ")"
    allocate_clause: ALLOCATE "(" allocate_parameter ")"
    ALLOCATE: "allocate"
    allocate_parameter: [const_num_or_var] var_list

    target_update_construct: target_update_directive
    target_update_directive: TARGET UPDATE target_update_clause*
    target_update_clause: motion_clause
                        | device_clause
                        | if_clause
    motion_clause: update_motion_type "(" variable_array_section_list ")"
    variable_array_section_list: PYTHON_NAME
                           //    | array_section
                               | name_slice
                               | variable_array_section_list "," PYTHON_NAME
                               | variable_array_section_list "," name_slice
                           //    | variable_array_section_list "," array_section
    //array_section: PYTHON_NAME array_section_subscript
    //array_section_subscript: array_section_subscript "[" [const_num_or_var] ":" [const_num_or_var] "]"
    //                       | array_section_subscript "[" const_num_or_var "]"
    //                       | "[" [const_num_or_var] ":" [const_num_or_var] "]"
    //                       | "[" const_num_or_var "]"
    TARGET: "target"
    TEAMS: "teams"
    DISTRIBUTE: "distribute"
    single_construct: single_directive
    single_directive: SINGLE [single_clause*]
    SINGLE: "single"
    single_clause: unique_single_clause
                 | private_clause
                 | firstprivate_clause
                 | NOWAIT
    unique_single_clause: copyprivate_clause
    NOWAIT: "nowait"
    master_construct: master_directive
    master_directive: "master"
    simd_construct: simd_directive
    simd_directive: SIMD [simd_clause*]
    SIMD: "simd"
    simd_clause: collapse_clause
               | aligned_clause
               | linear_clause
               | uniform_clause
               | reduction_clause
               | inbranch_clause
    aligned_clause: ALIGNED "(" var_list ")"
                  | ALIGNED "(" var_list ":" const_num_or_var ")"
    declare_simd_construct: declare_simd_directive_seq
    declare_simd_directive_seq: declare_simd_directive
                              | declare_simd_directive_seq declare_simd_directive
    declare_simd_directive: SIMD [declare_simd_clause*]
    declare_simd_clause: "simdlen" "(" const_num_or_var ")"
                       | aligned_clause
                       | linear_clause
                       | uniform_clause
                       | reduction_clause
                       | inbranch_clause
    ALIGNED: "aligned"
    inbranch_clause: INBRANCH | NOTINBRANCH
    INBRANCH: "inbranch"
    NOTINBRANCH: "notinbranch"
    uniform_clause: UNIFORM "(" var_list ")"
    UNIFORM: "uniform"
    collapse_clause: COLLAPSE "(" const_num_or_var ")"
    COLLAPSE: "collapse"
    task_construct: task_directive
    TASK: "task"
    task_directive: TASK [task_clause*]
    task_clause: unique_task_clause
               | data_sharing_clause
               | private_clause
               | firstprivate_clause
               | data_default_clause
    unique_task_clause: if_clause
                      | UNTIED
                      | MERGEABLE
                      | FINAL "(" const_num_or_var ")"
                      | depend_with_modifier_clause
    DEPEND: "depend"
    FINAL: "final"
    UNTIED: "untied"
    MERGEABLE: "mergeable"
    dependence_type: IN
                   | OUT
                   | INOUT
    depend_with_modifier_clause: DEPEND "(" dependence_type ":" variable_array_section_list ")"
    IN: "in"
    OUT: "out"
    INOUT: "inout"
    data_default_clause: default_shared_clause
                       | default_none_clause
    data_sharing_clause: shared_clause
    data_clause: private_clause
               | copyprivate_clause
               | firstprivate_clause
               | lastprivate_clause
               | data_sharing_clause
               | data_default_clause
               | copyin_clause
               | reduction_clause
    private_clause: PRIVATE "(" var_list ")"
    PRIVATE: "private"
    copyprivate_clause: COPYPRIVATE "(" var_list ")"
    COPYPRIVATE: "copyprivate"
    firstprivate_clause: FIRSTPRIVATE "(" var_list ")"
    FIRSTPRIVATE: "firstprivate"
    lastprivate_clause: LASTPRIVATE "(" var_list ")"
    LASTPRIVATE: "lastprivate"
    shared_clause: SHARED "(" var_list ")"
    SHARED: "shared"
    copyin_clause: COPYIN "(" var_list ")"
    COPYIN: "copyin"
    REDUCTION: "reduction"
    DEFAULT: "default"
    reduction_clause: REDUCTION "(" reduction_operator ":" var_list ")"
    default_shared_clause: DEFAULT "(" "shared" ")"
    default_none_clause: DEFAULT "(" "none" ")"
    const_num_or_var: NUMBER | PYTHON_NAME
    parallel_construct: parallel_directive
    parallel_directive: PARALLEL [parallel_clause*]
    parallel_clause: unique_parallel_clause
                   | data_default_clause
                   | private_clause
                   | firstprivate_clause
                   | data_sharing_clause
                   | reduction_clause
    unique_parallel_clause: if_clause | num_threads_clause
    if_clause: IF "(" const_num_or_var ")"
    IF: "if"
    num_threads_clause: NUM_THREADS "(" const_num_or_var ")"
    NUM_THREADS: "num_threads"
    PARALLEL: "parallel"
    FOR: "for"
    parallel_for_construct: parallel_for_directive
    parallel_for_directive: PARALLEL FOR [parallel_for_clause*]
    parallel_for_clause: unique_parallel_clause
                       | unique_for_clause
                       | data_default_clause
                       | private_clause
                       | firstprivate_clause
                       | lastprivate_clause
                       | data_sharing_clause
                       | reduction_clause
    for_construct: for_directive
    for_directive: FOR [for_clause*]
    for_clause: unique_for_clause | data_clause | NOWAIT
    unique_for_clause: ORDERED
                     | schedule_clause
                     | collapse_clause
    LINEAR: "linear"
    linear_clause: LINEAR "(" var_list ":" const_num_or_var ")"
                 | LINEAR "(" var_list ")"
    sched_no_expr: SCHEDULE "(" schedule_kind ")"
    sched_expr: SCHEDULE "(" schedule_kind "," const_num_or_var ")"
    schedule_clause: sched_no_expr
                   | sched_expr
    SCHEDULE: "schedule"
    schedule_kind: STATIC | DYNAMIC | GUIDED | RUNTIME | AUTO
    STATIC: "static"
    DYNAMIC: "dynamic"
    GUIDED: "guided"
    RUNTIME: "runtime"
    AUTO: "auto"
    COLON: ":"
    oslice: [const_num_or_var] COLON [const_num_or_var]
    slice_list: oslice | slice_list "," oslice
    name_slice: PYTHON_NAME [ "[" slice_list "]" ]
    var_list: name_slice | var_list "," name_slice
    number_list: NUMBER | number_list "," NUMBER
    PLUS: "+"
    MINUS: "-"
    STAR: "*"
    reduction_operator: PLUS | "\\" | STAR | MINUS | "&" | "^" | "|" | "&&" | "||"
    threadprivate_directive: "threadprivate" "(" var_list ")"
    cancellation_point_directive: "cancellation point" construct_type_clause
    construct_type_clause: PARALLEL
                         | SECTIONS
                         | FOR
                         | TASKGROUP
    cancel_directive: "cancel" construct_type_clause [if_clause]
    ordered_directive: ORDERED
    ordered_construct: ordered_directive
    ORDERED: "ordered"
    flush_directive: "flush" "(" var_list ")"

    region_phrase: "(" PYTHON_NAME ")"
    PYTHON_NAME: /[a-zA-Z_]\w*/

    %import common.NUMBER
    %import common.WS
    %ignore WS
    """

"""
    name_slice: PYTHON_NAME [ "[" slice ["," slice]* "]" ]
"""
