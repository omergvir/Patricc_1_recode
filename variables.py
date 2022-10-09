count_features = ['Child affect', 'Child affect:positive 1', 'Child affect:positive 2', 'Child affect:positive 3',
            'Child affective touch:affective touch', 'Child gaze:parent', 'Child gaze:props',
            'Child gaze:robot', 'Child gesture:point at prop', 'Child prop manipulation:child',
            'Child utterance:utterance',
            'Conversational turns', 'Conversational turns:CP', 'Conversational turns:CPC', 'Conversational turns:PC',
            'Conversational turns:PCP', 'Joint attention', 'Joint attention:props', 'Joint attention:robot',
            'Mutual gaze:MG',
            'Non-verbal scaffolding', 'Non-verbal scaffolding:cognitive', 'Non-verbal scaffolding:affective',
            'Non-verbal scaffolding:technical',
            'Verbal scaffolding', 'Verbal scaffolding:affective', 'Verbal scaffolding:cognitive',
            'Verbal scaffolding:technical',
            'Parent affect', 'Parent affect:positive 1', 'Parent affect:positive 2', 'Parent affect:positive 3',
            'Parent affective touch:affective touch', 'Parent gesture:point at prop', 'Parent prop manipulation:parent',
            'Parent gaze:child', 'Parent gaze:props', 'Parent gaze:robot', 'Parent utterance:utterance']

robot_features = ['robot text:positive feedback','robot text:pick up',
                  'robot pointing:point at prop']

granger_features = count_features + robot_features

granger_condition_list = [['Non-verbal scaffolding:affective','robot text:positive feedback'],
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

granger_robot_tests = [['Child gaze:props', 'robot pointing:point at prop'],
                 ['Parent gaze:props', 'robot pointing:point at prop'],
                 ['Joint attention:props', 'robot pointing:point at prop'],
                 ['Non-verbal scaffolding:affective','robot text:positive feedback'],
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



