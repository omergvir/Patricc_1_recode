count_features_old = ['Child affect', 'Child affect:positive 1', 'Child affect:positive 2', 'Child affect:positive 3',
                  'Child affective touch:affective touch', 'Child gaze:parent', 'Child gaze:props', 'Child gaze:tablet',
                  'Child gaze:robot', 'Child gesture:point at prop', 'Child prop manipulation:child',
                  'Child utterance:utterance',
                  'Conversational turns', 'Conversational turns:CP', 'Conversational turns:CPC',
                  'Conversational turns:PC',
                  'Conversational turns:PCP', 'Joint attention', 'Joint attention:props', 'Joint attention:robot',
                  'Mutual gaze:MG',
                  'Non-verbal scaffolding', 'Non-verbal scaffolding:cognitive', 'Non-verbal scaffolding:affective',
                  'Non-verbal scaffolding:technical',
                  'Verbal scaffolding', 'Verbal scaffolding:affective', 'Verbal scaffolding:cognitive',
                  'Verbal scaffolding:technical',
                  'Parent affect', 'Parent affect:positive 1', 'Parent affect:positive 2', 'Parent affect:positive 3',
                  'Parent affective touch:affective touch', 'Parent gesture:point at prop',
                  'Parent prop manipulation:parent',
                  'Parent gaze:child', 'Parent gaze:props', 'Parent gaze:robot', 'Parent gaze:tablet',
                  'Parent utterance:utterance']

count_features = ['Child affect',
                  'Child affective touch:affective touch', 'Child gaze:parent', 'Child gaze:props', 'Child gaze:tablet',
                  'Child gaze:robot', 'Child prop manipulation:child',
                  'Child utterance:utterance',
                  'Conversational turns',
                  'Mutual gaze:MG',
                  'Non-verbal scaffolding:cognitive', 'Non-verbal scaffolding:affective',
                  'Non-verbal scaffolding:technical',
                  'Verbal scaffolding:affective', 'Verbal scaffolding:cognitive',
                  'Verbal scaffolding:technical',
                  'Parent affect',
                  'Parent affective touch:affective touch',
                  'Parent prop manipulation:parent',
                  'Parent gaze:child', 'Parent gaze:props', 'Parent gaze:robot', 'Parent gaze:tablet',
                  'Parent utterance:utterance']

robot_vs_tablet_old = ['Child gaze:parent', 'Parent gaze:child', 'Mutual gaze:MG',
                   'Child gaze:object', 'Parent gaze:object',
                   'Parent affective touch:affective touch', 'Child affective touch:affective touch',
                   'Child affect', 'Parent affect',
                   'Conversational turns','Child utterance:utterance', 'Parent utterance:utterance',
                   'Non-verbal scaffolding:cognitive', 'Non-verbal scaffolding:affective',
                   'Non-verbal scaffolding:technical',
                   'Verbal scaffolding:affective', 'Verbal scaffolding:cognitive', 'Verbal scaffolding:technical']

robot_vs_tablet = ['Child gaze:parent_normalized total time', 'Parent gaze:child_normalized total time',
                   'Child gaze:object_normalized total time', 'Parent gaze:object_normalized total time',
                   'Mutual gaze:MG',
                   'Parent affective touch:affective touch_normalized count',
                   'Child affective touch:affective touch_normalized count',
                   'Child affect_normalized total time', 'Parent affect_normalized total time',
                   'Conversational turns_normalized count',
                   'Child utterance:utterance_normalized count',
                   'Parent utterance:utterance_normalized count',
                   'Non-verbal scaffolding:cognitive_normalized count',
                   'Non-verbal scaffolding:affective_normalized count',
                   'Non-verbal scaffolding:technical_normalized count',
                   'Verbal scaffolding:affective_normalized count',
                   'Verbal scaffolding:cognitive_normalized count',
                   'Verbal scaffolding:technical_normalized count', 'Mutual gaze:MG']

robot_features = ['robot text:positive feedback', 'robot text:pick up',
                  'robot pointing:point at prop']

granger_features = count_features + robot_features

granger_condition_list = [['Non-verbal scaffolding:affective', 'robot text:positive feedback'],
                          ['Verbal scaffolding:affective', 'robot text:positive feedback'],
                          ['Parent affective touch:affective touch', 'robot text:positive feedback'],
                          ['Parent affect', 'robot text:positive feedback'],
                          ['Child affect', 'robot text:positive feedback'],
                          ['Mutual gaze:MG', 'robot text:pick up'],
                          ['Child gaze:object', 'robot text:pick up'],
                          ['Parent gaze:object', 'robot text:pick up'],
                          ['Parent utterance:utterance', 'robot text:pick up'],
                          ['Child utterance:utterance', 'robot text:pick up'],
                          ['Conversational turns', 'robot text:pick up'],
                          ['Non-verbal scaffolding:cognitive', 'robot text:pick up'],
                          ['Verbal scaffolding:cognitive', 'robot text:pick up'],
                          ['Parent prop manipulation:parent', 'robot text:pick up'],
                          ['Child prop manipulation:child', 'robot text:pick up']]

granger_robot_tests = [['Child gaze:props', 'robot pointing:point at prop'],
                       ['Parent gaze:props', 'robot pointing:point at prop'],
                       ['Joint attention:props', 'robot pointing:point at prop'],
                       ['Non-verbal scaffolding:affective', 'robot text:positive feedback'],
                       ['Verbal scaffolding:affective', 'robot text:positive feedback'],
                       ['Parent affective touch:affective touch', 'robot text:positive feedback'],
                       ['Parent affect', 'robot text:positive feedback'],
                       ['Child affect', 'robot text:positive feedback'],
                       ['Mutual gaze:MG', 'robot text:pick up'],
                       ['Child gaze:robot', 'robot text:pick up'],
                       ['Parent gaze:robot', 'robot text:pick up'],
                       ['Parent utterance:utterance', 'robot text:pick up'],
                       ['Child utterance:utterance', 'robot text:pick up'],
                       ['Conversational turns', 'robot text:pick up'],
                       ['Non-verbal scaffolding:cognitive', 'robot text:pick up'],
                       ['Verbal scaffolding:cognitive', 'robot text:pick up'],
                       ['Parent prop manipulation:parent', 'robot text:pick up'],
                       ['Child prop manipulation:child', 'robot text:pick up'],
                       ['Parent gesture:point at prop', 'robot text:pick up'],
                       ['Child gesture:point at prop', 'robot text:pick up']]


additional_features = ['Conversational turns', 'Conversational turns:CP', 'Conversational turns:CPC',
                       'Conversational turns:PC', 'Conversational turns:PCP', 'Joint attention',
                       'Joint attention:props', 'Joint attention:robot']

time_series_features = count_features + robot_features + additional_features

