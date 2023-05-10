import pickle

from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, render_template, url_for, redirect
from cooking_ner import NER_Document
from flask import make_response

app = Flask(__name__)

app.config['SECRET_KEY'] = 'fc3bb2a43ff1103895a4ee315ee27740'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_users.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

doc = None


class Entity(db.Model):
    name = db.Column(db.String(50), unique=True, primary_key=True)
    frequency = db.Column(db.Integer, default=0)


class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    entities = db.Column(db.String(1000), nullable=False)
    ner_doc = db.Column(db.LargeBinary, nullable=False)

    # Add methods to serialize and deserialize the NER_Document object
    def set_ner_doc(self, ner_doc):
        self.ner_doc = pickle.dumps(ner_doc)

    def get_ner_doc(self):
        return pickle.loads(self.ner_doc)


def create_all():
    with app.app_context():
        # db.drop_all()
        db.create_all()


create_all()


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/submit', methods=['POST'])
def form_submit():
    return redirect(url_for('form'))


@app.route('/', methods=['POST'])
def form_post():
    global doc
    text = request.form.get('text', '')
    doc = NER_Document(text)
    entities_markup = doc.get_entities_with_markup()

    # Create a new annotation when the user submits the form
    annotation = Annotation(input_text=doc.text, entities=str(doc.entities))
    annotation.set_ner_doc(doc)
    db.session.add(annotation)
    db.session.commit()
    annotation_id = annotation.id  # Store the annotation ID for the result page

    for entity in doc.entities:
        db_entity = Entity.query.filter_by(name=entity.text).first()
        if db_entity:
            db_entity.frequency += 1
        else:
            db_entity = Entity(name=entity.text, frequency=1)
            db.session.add(db_entity)
        db.session.commit()

    return render_template('result.html', entities_markup=entities_markup, text=doc.text, entities=doc.entities,
                           annotation_id=annotation_id)


@app.route('/update', methods=['POST'])
def update():
    global doc
    annotation_id = request.form.get("annotation_id")

    if annotation_id:
        annotation = Annotation.query.get(int(annotation_id))
        doc = annotation.get_ner_doc()

    if doc is None:
        return redirect(url_for('form'))

    if request.method == 'POST':
        for entity in doc.entities:
            new_label = request.form.get(f"entity_update_{entity.id}")
            if new_label is not None:
                entity.label = new_label
        annotation.set_ner_doc(doc)
        db.session.commit()

        # Re-fetch the updated data
        doc = annotation.get_ner_doc()
        entities_markup = doc.get_entities_with_markup()

    return render_template('result.html', entities_markup=entities_markup, text=doc.text, entities=doc.entities,
                           annotation_id=annotation_id)


@app.route('/entities', methods=['GET', 'POST'])
def entities():
    annotations = Annotation.query.all()
    return render_template('entities.html', annotations=annotations)

@app.route('/export', methods=['POST'])
def export():
    global doc
    output = ""
    for entity in doc.entities:
        output += f"{entity.text} - {entity.label}\n"

    # Generate the plain text
    filename = 'output.txt'  # Specify the desired output file name
    text = "NER Output:\n\n" + output

    # Create a response with the generated file
    response = make_response()

    # Export as plain text
    response.data = text
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    response.headers['Content-type'] = 'text/plain'

    return response


@app.route('/edit_annotation/<int:annotation_id>', methods=['GET', 'POST'])
def edit_annotation(annotation_id):
    annotation = Annotation.query.get(annotation_id)
    doc = annotation.get_ner_doc()
    entities_markup = doc.get_entities_with_markup()

    return render_template(
        'result.html',
        entities_markup=entities_markup,
        text=doc.text,
        entities=doc.entities,
        annotation_id=annotation_id,
    )


@app.route('/delete_annotation/<int:annotation_id>', methods=['POST'])
def delete_annotation(annotation_id):
    annotation = Annotation.query.get(annotation_id)
    if annotation:
        db.session.delete(annotation)
        db.session.commit()
        return redirect(url_for('entities'))
    return "Error: No annotation found for ID: {}".format(annotation_id)




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

