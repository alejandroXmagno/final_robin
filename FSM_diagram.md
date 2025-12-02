# Autonomous Exploration Robot - Finite State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INITIALIZATION                                     │
│  - Wait for nav2 action server (30s timeout)                               │
│  - Initialize MediaPipe BlazePose                                           │
│  - Subscribe to /map, /camera/*, /scan                                      │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ WAITING_FOR_MAP │ ◄───────────────┐
                    │                 │                  │
                    │ current_map     │                  │
                    │ == None         │                  │
                    └────────┬────────┘                  │
                             │                           │
                             │ map received              │
                             ▼                           │
              ┌──────────────────────────┐              │
              │ WAITING_FOR_MAP_FRAME    │              │
              │                          │              │
              │ TF: map->base_link       │              │
              │ not available            │              │
              └──────────┬───────────────┘              │
                         │                              │
                         │ map frame available          │
                         ▼                              │
        ┌────────────────────────────────┐             │
        │           IDLE                 │             │
        │                                │             │
        │ - No active navigation goal    │ ◄───────┐  │
        │ - Ready to explore             │         │  │
        │ - Checking for waving person   │         │  │
        │   (every 0.2s)                 │         │  │
        └──────┬─────────────┬───────────┘         │  │
               │             │                     │  │
    waving     │             │ no waving detected  │  │
    person     │             │ + frontier found    │  │
    detected   │             ▼                     │  │
    (dist >    │   ┌─────────────────────┐        │  │
    0.3m)      │   │ NAVIGATING_TO       │        │  │
               │   │ FRONTIER            │        │  │
               │   │                     │        │  │
               │   │ - Active nav goal   │        │  │
               │   │ - Following path    │        │  │
               │   │ - Checking for      │        │  │
               │   │   waving person     │        │  │
               │   │   (overrides goal)  │        │  │
               │   └──┬────────┬─────────┘        │  │
               │      │        │                  │  │
               │      │        │ goal succeeded   │  │
               │      │        └──────────────────┘  │
               │      │                              │
               │      │ timeout (120s)               │
               │      │ OR goal rejected             │
               │      └──────────────────────────────┤
               │                                     │
               │                                     │
               ▼                                     │
    ┌──────────────────────┐                        │
    │  APPROACHING_PERSON  │                        │
    │                      │                        │
    │  - Cancel current    │                        │
    │    exploration       │                        │
    │  - Navigate to       │                        │
    │    person location   │                        │
    │    (stop 1 ft away)  │                        │
    └──────┬────────┬──────┘                        │
           │        │                               │
           │        │ reached person                │
           │        │ OR already close              │
           │        ▼                               │
           │  ┌─────────────────┐                  │
           │  │ WAITING_NEAR    │                  │
           │  │ PERSON          │                  │
           │  │                 │                  │
           │  │ - 10 second     │                  │
           │  │   wait timer    │                  │
           │  │ - No exploring  │                  │
           │  └────────┬────────┘                  │
           │           │                           │
           │           │ 10 seconds elapsed        │
           │           └───────────────────────────┘
           │
           │ goal rejected
           │ OR nav failed
           └───────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                    CONCURRENT BACKGROUND PROCESSES
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  POSE DETECTION (runs at 5 Hz, parallel to main FSM)                       │
│                                                                             │
│  IDLE ──RGB frame──► DETECTING ──pose found──► CHECK_WAVING ──yes──► UPDATE│
│   ▲                      │                          │             waving   │
│   │                      │                          │             flag     │
│   │                      │ no pose                  │ no                   │
│   └──────────────────────┴──────────────────────────┴──────────────────────┘
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  TIMEOUT CHECKER (runs at 1 Hz, parallel to main FSM)                      │
│                                                                             │
│  CHECKING ──goal active──► CHECK_TIME ──> 120s elapsed──► CANCEL_GOAL      │
│      ▲                          │                                           │
│      │                          │ < 120s                                    │
│      └──────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FRONTIER EXPLORATION TIMER (runs every 3 seconds)                          │
│                                                                             │
│  ┌──► WAITING ──timer──► CHECK_STATE ──can explore──► FIND_FRONTIERS       │
│  │       ▲                    │                            │                │
│  │       │                    │ busy/approaching           │                │
│  │       └────────────────────┘                            │                │
│  │                                                          ▼                │
│  │                                                   CLUSTER_FRONTIERS       │
│  │                                                          │                │
│  │                                                          ▼                │
│  │                                                   SELECT_BEST             │
│  │                                                          │                │
│  │                                                          ▼                │
│  └──────────────────────────────────────────────────SEND_NAV_GOAL           │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                            STATE VARIABLES
═══════════════════════════════════════════════════════════════════════════════

• approaching_person: bool           - True when navigating to/waiting at person
• person_wait_start_time: float      - Timestamp when reached person (None = not waiting)
• navigation_goal_handle: GoalHandle - Active navigation goal (None = idle)
• exploring: bool                    - Global enable flag for exploration
• map_frame_available: bool          - TF map frame exists
• is_waving: bool                    - Person waving detected
• current_map: OccupancyGrid         - Latest SLAM map
• current_target: (x, y)             - Current navigation target position


═══════════════════════════════════════════════════════════════════════════════
                         PRIORITY & PREEMPTION
═══════════════════════════════════════════════════════════════════════════════

                        HIGHEST PRIORITY
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Person Detection             │
              │  (preempts exploration)       │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Goal Timeout                 │
              │  (safety mechanism)           │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Frontier Exploration         │
              │  (default behavior)           │
              └───────────────────────────────┘
                              │
                              ▼
                        LOWEST PRIORITY

```

## State Descriptions

### Main States

1. **INITIALIZATION**
   - Entry: Program start
   - Actions: Initialize all nodes, wait for action servers
   - Exit: Nav2 ready OR 30s timeout

2. **WAITING_FOR_MAP**
   - Entry: No map data received
   - Actions: Log waiting status every cycle
   - Exit: First map message received

3. **WAITING_FOR_MAP_FRAME**
   - Entry: Map exists but TF not available
   - Actions: Attempt TF lookup every 0.1s
   - Exit: TF map→base_link lookup succeeds

4. **IDLE**
   - Entry: No active navigation goal
   - Actions: 
     - Check for waving person (0.2s interval)
     - Wait for exploration timer (3s interval)
   - Exit: Waving detected OR frontier selected

5. **NAVIGATING_TO_FRONTIER**
   - Entry: Nav goal sent to frontier
   - Actions:
     - Follow planned path
     - Monitor for waving person (can preempt)
     - Check timeout (120s max)
   - Exit: Goal reached, timeout, or preempted

6. **APPROACHING_PERSON**
   - Entry: Waving person detected
   - Actions:
     - Cancel current exploration goal
     - Navigate to position 1 ft from person
     - Orient toward person
   - Exit: Reached person OR navigation failed

7. **WAITING_NEAR_PERSON**
   - Entry: Arrived near person
   - Actions:
     - Start 10-second timer
     - Maintain position
     - No exploration
   - Exit: Timer expires (10s)

## Transition Conditions

| From State | To State | Condition |
|------------|----------|-----------|
| INITIALIZATION | WAITING_FOR_MAP | Nav2 ready |
| WAITING_FOR_MAP | WAITING_FOR_MAP_FRAME | Map received |
| WAITING_FOR_MAP_FRAME | IDLE | TF available |
| IDLE | NAVIGATING_TO_FRONTIER | Frontier selected & no waving |
| IDLE | APPROACHING_PERSON | Waving detected & dist > 0.3m |
| NAVIGATING_TO_FRONTIER | IDLE | Goal succeeded |
| NAVIGATING_TO_FRONTIER | APPROACHING_PERSON | Waving detected |
| NAVIGATING_TO_FRONTIER | IDLE | Timeout (120s) |
| APPROACHING_PERSON | WAITING_NEAR_PERSON | Reached person |
| APPROACHING_PERSON | IDLE | Navigation failed |
| WAITING_NEAR_PERSON | IDLE | 10 seconds elapsed |

## Key Decision Logic

```python
# Main state decision (simplified)
if approaching_person:
    if person_wait_start_time is not None:
        if elapsed >= 10.0:
            → IDLE
        else:
            → WAITING_NEAR_PERSON
    else:
        → APPROACHING_PERSON
elif is_waving and distance > 0.3m:
    → APPROACHING_PERSON
elif navigation_goal_handle is not None:
    if timeout:
        → IDLE
    else:
        → NAVIGATING_TO_FRONTIER
elif frontiers_available and map_frame_available:
    → NAVIGATING_TO_FRONTIER
else:
    → IDLE (or WAITING states)
```

