import ios_operation

def action_bounties(action):
    action.phone_action([]) #select 
    action.phone_action([]) #confirm
    
def action_team(action):
    action.phone_action([]) #select 
    action.phone_action([]) #confirm
    action.phone_action([]) #pop

def action_map(action, location):
    action.phone_action(location) #select 
    action.phone_action([]) #confirm
    
def action_battle(action, location):
    pass

def action_treassure(action):
    action.phone_action([]) #select 
    action.phone_action([]) #confirm
    
def action_chest(action):
    action.phone_action([]) #select 
    action.phone_action([]) #select 
    action.phone_action([]) #select 
    action.phone_action([]) #confirm
    
    
    
    