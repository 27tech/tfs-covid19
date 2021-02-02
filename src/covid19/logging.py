from logging import config

LOG_LEVEL = 'DEBUG'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{asctime} {levelname} |{name}| {message}',
            'style': '{',
        }

    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
    },
    'loggers': {
        'covid19': {
            'propagate': True,
            'handlers': ['console'],
            'level': LOG_LEVEL,
        },
    }
}


def configure_logging():
    config.dictConfig(LOGGING)
