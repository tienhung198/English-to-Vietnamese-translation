from modules import *
from flask import Flask, request, jsonify, render_template
import torch

app = Flask(__name__)

# Tạo mô hình và bộ tối ưu hóa với cùng các tham số như trước đó
model = Transformer(len(SRC.vocab), len(TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']).to(opt['device'])

optimizer = ScheduledOptim(
    torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    0.2, opt['d_model'], 4000
)

# Tải checkpoint
checkpoint = torch.load('model_dichmay/model_dichmay.pth', map_location=opt['device'])

# Tải trạng thái của mô hình và bộ tối ưu hóa
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Endpoint chính cho trang web
@app.route('/', methods=['GET'])
def index():
    return render_template('index2.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    original_sentence = data.get('sentence', '')
    translation = ""
    if original_sentence:
        translation = translate_sentence(original_sentence, model, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])
        translation = translation.replace('_', ' ').replace('#', ' ')
    
    return jsonify({'translation': translation})

if __name__ == '__main__':
    app.run(debug=True)
