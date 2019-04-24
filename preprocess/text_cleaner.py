'''
	Copyright (c) 2019 Aria-K-Alethia@github.com / xindetai@Beihang University

	Description:
		text cleaner for removing noise in text
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import re

class TextCleaner(object):
	"""
		overview:
			text cleaner for removing noise in text
	"""
	def __init__(self):
		super(TextCleaner, self).__init__()
		self.url = re.compile((
				r'http[s]?://(?:[a-zA-Z]|'
				r'[0-9]|[$-_@.&+]|[!*\(\),]|'
				r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'))
		self.mail = re.compile(r'([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})*')
		self.html_tag = re.compile(r'<\/?[\w\s]*>|<.+[\W]>')
		self.i18n_tel = re.compile((r'(?:(?:\(?(?:00|\+)([1-4]\d\d|'
								r'[1-9]\d?)\)?)?[\-\.\ \\\/]?)?((?:\(?\d{1,}\)?'
								r'[\-\.\ \\\/]?){0,})(?:[\-\.\ \\\/]?(?:#|ext\.?|'
								r'extension|x)[\-\.\ \\\/]?(\d+))?'))