"""Plant-side contact classification shared by recording and offline evaluation."""


def is_static_body(model, body):
    """A fixed child of a free object is moving, despite having no own joint."""
    while body:
        if model.body_jntnum[body]:
            return False
        body = int(model.body_parentid[body])
    return True
