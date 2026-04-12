from conf.entities.skills.tapas_skills import TapasSkillSet


BASE = [
    TapasSkillSet.close_drawer,
    TapasSkillSet.close_drawer_back,
    TapasSkillSet.open_drawer,
    TapasSkillSet.open_drawer_back,
    TapasSkillSet.press_button,
    TapasSkillSet.press_button_back,
]

SLIDE = [
    TapasSkillSet.open_slide,
    TapasSkillSet.close_slide,
    TapasSkillSet.open_slide_back,
    TapasSkillSet.close_slide_back,
]

RED = [
    TapasSkillSet.pick_red_table,
    TapasSkillSet.place_red_table,
    TapasSkillSet.pick_red_drawer,
    TapasSkillSet.place_red_drawer,
]

PINK = [
    TapasSkillSet.pick_pink_table,
    TapasSkillSet.place_pink_table,
    TapasSkillSet.pick_pink_drawer,
    TapasSkillSet.place_pink_drawer,
]

BLUE = [
    TapasSkillSet.pick_blue_table,
    TapasSkillSet.place_blue_table,
    TapasSkillSet.pick_blue_drawer,
    TapasSkillSet.place_blue_drawer,
]


SKILL_SETS = {
    "base": BASE,
    "slider": BASE + SLIDE,
    "red": BASE + RED,
    "pink": BASE + PINK,
    "blue": BASE + BLUE,
    "sr": BASE + SLIDE + RED,
    "srp": BASE + SLIDE + RED + PINK,
    "srpb": BASE + SLIDE + RED + PINK + BLUE,
}
# if storage.config.skills == SET_SLIDE:
#          num_skills = 6
#      elif (
#          storage.config.skills == SET_RED
#           or storage.config.skills == SET_BLUE
#           or storage.config.skills == SET_PINK
#        ):
#            num_skills = 8
#        elif storage.config.skills == SET_SR:
#            num_skills = 10
#        elif storage.config.skills == SET_SRP:
#            num_skills = 12
#        elif storage.config.skills == SET_SRPB:
#            num_skills = 14
