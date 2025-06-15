from datetime import datetime, timedelta


def get_simple_mock_data() -> dict:
    """
    Return simple hardcoded mock data for testing quality reports.

    Much simpler than the original complex randomization - just enough
    data to fill out all sections of the quality report for testing.

    Returns:
        Dictionary matching the structure expected by QualityMetricsAnalyzer
    """

    # Simple hardcoded timestamps
    base_time = datetime.now() - timedelta(hours=2)

    return {
        'dataset_info': {
            'total_participants': 4,
            'start_time': base_time.isoformat(),
            'end_time': (base_time + timedelta(hours=2)).isoformat(),
            'completed_participants': 3
        },
        'participants': {
            'S001F': {
                'start_time': base_time.isoformat(),
                'end_time': (base_time + timedelta(minutes=45)).isoformat(),
                'completed': True,
                'conditions': {
                    'Baseline': {
                        'start_time': base_time.isoformat(),
                        'completion': {
                            'success': True,
                            'error': None,
                            'end_time': (base_time + timedelta(minutes=15)).isoformat()
                        },
                        'stages': {
                            'filter': {
                                'metrics': {'filter_applied': True, 'highpass': 1.0, 'lowpass': 40.0},
                                'timestamp': base_time.isoformat()
                            },
                            'detect_bad_channels': {
                                'metrics': {
                                    'n_original': 1,
                                    'n_detected': 3,
                                    'n_final': 2,
                                    'original_bads': ['EEG12'],
                                    'detected_bads': ['EEG23', 'EEG45', 'EEG67'],
                                    'interpolation_successful': True
                                },
                                'timestamp': (base_time + timedelta(minutes=2)).isoformat()
                            },
                            'rereference': {
                                'metrics': {'rereference_applied': True, 'reference_type': 'average'},
                                'timestamp': (base_time + timedelta(minutes=4)).isoformat()
                            },
                            'blink_artifact': {
                                'metrics': {
                                    'n_components_excluded': 2,
                                    'excluded_components': [0, 1],
                                    'ica_applied': True
                                },
                                'timestamp': (base_time + timedelta(minutes=6)).isoformat()
                            },
                            'epoch': {
                                'metrics': {
                                    'total_epochs': 100,
                                    'kept_epochs': 85,
                                    'rejected_epochs': 15,
                                    'rejection_rate': 15.0,
                                    'rejection_reasons': {'amplitude': 10, 'muscle': 5}
                                },
                                'timestamp': (base_time + timedelta(minutes=8)).isoformat()
                            }
                        }
                    },
                    'Task1': {
                        'start_time': (base_time + timedelta(minutes=15)).isoformat(),
                        'completion': {
                            'success': True,
                            'error': None,
                            'end_time': (base_time + timedelta(minutes=30)).isoformat()
                        },
                        'stages': {
                            'filter': {
                                'metrics': {'filter_applied': True, 'highpass': 1.0, 'lowpass': 40.0},
                                'timestamp': (base_time + timedelta(minutes=15)).isoformat()
                            },
                            'detect_bad_channels': {
                                'metrics': {
                                    'n_original': 0,
                                    'n_detected': 2,
                                    'n_final': 1,
                                    'original_bads': [],
                                    'detected_bads': ['EEG34', 'EEG56'],
                                    'interpolation_successful': True
                                },
                                'timestamp': (base_time + timedelta(minutes=17)).isoformat()
                            },
                            'rereference': {
                                'metrics': {'rereference_applied': True, 'reference_type': 'average'},
                                'timestamp': (base_time + timedelta(minutes=19)).isoformat()
                            },
                            'blink_artifact': {
                                'metrics': {
                                    'n_components_excluded': 3,
                                    'excluded_components': [0, 2, 5],
                                    'ica_applied': True
                                },
                                'timestamp': (base_time + timedelta(minutes=21)).isoformat()
                            },
                            'epoch': {
                                'metrics': {
                                    'total_epochs': 120,
                                    'kept_epochs': 108,
                                    'rejected_epochs': 12,
                                    'rejection_rate': 10.0,
                                    'rejection_reasons': {'amplitude': 8, 'muscle': 4}
                                },
                                'timestamp': (base_time + timedelta(minutes=23)).isoformat()
                            }
                        }
                    }
                }
            },
            'S002M': {
                'start_time': (base_time + timedelta(minutes=50)).isoformat(),
                'end_time': (base_time + timedelta(minutes=90)).isoformat(),
                'completed': True,
                'conditions': {
                    'Baseline': {
                        'start_time': (base_time + timedelta(minutes=50)).isoformat(),
                        'completion': {
                            'success': True,
                            'error': None,
                            'end_time': (base_time + timedelta(minutes=65)).isoformat()
                        },
                        'stages': {
                            'filter': {
                                'metrics': {'filter_applied': True, 'highpass': 1.0, 'lowpass': 40.0},
                                'timestamp': (base_time + timedelta(minutes=50)).isoformat()
                            },
                            'detect_bad_channels': {
                                'metrics': {
                                    'n_original': 0,
                                    'n_detected': 5,
                                    'n_final': 3,
                                    'original_bads': [],
                                    'detected_bads': ['EEG11', 'EEG22', 'EEG33', 'EEG44', 'EEG55'],
                                    'interpolation_successful': True
                                },
                                'timestamp': (base_time + timedelta(minutes=52)).isoformat()
                            },
                            'rereference': {
                                'metrics': {'rereference_applied': True, 'reference_type': 'average'},
                                'timestamp': (base_time + timedelta(minutes=54)).isoformat()
                            },
                            'blink_artifact': {
                                'metrics': {
                                    'n_components_excluded': 4,
                                    'excluded_components': [0, 1, 3, 7],
                                    'ica_applied': True
                                },
                                'timestamp': (base_time + timedelta(minutes=56)).isoformat()
                            },
                            'epoch': {
                                'metrics': {
                                    'total_epochs': 95,
                                    'kept_epochs': 76,
                                    'rejected_epochs': 19,
                                    'rejection_rate': 20.0,
                                    'rejection_reasons': {'amplitude': 12, 'muscle': 7}
                                },
                                'timestamp': (base_time + timedelta(minutes=58)).isoformat()
                            }
                        }
                    }
                }
            },
            'S003F': {
                'start_time': (base_time + timedelta(minutes=95)).isoformat(),
                'end_time': (base_time + timedelta(minutes=110)).isoformat(),
                'completed': True,
                'conditions': {
                    'Baseline': {
                        'start_time': (base_time + timedelta(minutes=95)).isoformat(),
                        'completion': {
                            'success': True,
                            'error': None,
                            'end_time': (base_time + timedelta(minutes=110)).isoformat()
                        },
                        'stages': {
                            'filter': {
                                'metrics': {'filter_applied': True, 'highpass': 1.0, 'lowpass': 40.0},
                                'timestamp': (base_time + timedelta(minutes=95)).isoformat()
                            },
                            'detect_bad_channels': {
                                'metrics': {
                                    'n_original': 0,
                                    'n_detected': 1,
                                    'n_final': 0,
                                    'original_bads': [],
                                    'detected_bads': ['EEG78'],
                                    'interpolation_successful': True
                                },
                                'timestamp': (base_time + timedelta(minutes=97)).isoformat()
                            },
                            'rereference': {
                                'metrics': {'rereference_applied': True, 'reference_type': 'average'},
                                'timestamp': (base_time + timedelta(minutes=99)).isoformat()
                            },
                            'blink_artifact': {
                                'metrics': {
                                    'n_components_excluded': 1,
                                    'excluded_components': [2],
                                    'ica_applied': True
                                },
                                'timestamp': (base_time + timedelta(minutes=101)).isoformat()
                            },
                            'epoch': {
                                'metrics': {
                                    'total_epochs': 110,
                                    'kept_epochs': 105,
                                    'rejected_epochs': 5,
                                    'rejection_rate': 4.5,
                                    'rejection_reasons': {'amplitude': 3, 'muscle': 2}
                                },
                                'timestamp': (base_time + timedelta(minutes=103)).isoformat()
                            }
                        }
                    }
                }
            },
            'S004M': {
                'start_time': (base_time + timedelta(minutes=115)).isoformat(),
                'end_time': (base_time + timedelta(minutes=115)).isoformat(),  # Same time = failed
                'completed': False,
                'conditions': {
                    'Baseline': {
                        'start_time': (base_time + timedelta(minutes=115)).isoformat(),
                        'completion': {
                            'success': False,
                            'error': "Stage 'filter' failed: Filter parameters invalid",
                            'end_time': (base_time + timedelta(minutes=115)).isoformat()
                        },
                        'stages': {
                            # Failed early, so only filter stage attempted
                            'filter': {
                                'metrics': {
                                    'error': 'Filter parameters invalid',
                                    'stage_completed': False,
                                    'timestamp': (base_time + timedelta(minutes=115)).isoformat()
                                },
                                'timestamp': (base_time + timedelta(minutes=115)).isoformat()
                            }
                        }
                    }
                }
            }
        }
    }