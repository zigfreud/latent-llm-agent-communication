import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from lip_protocol import LIPPacket

DEVICE = "cpu"
ADAPTER_PATH = os.path.join("experiments", "experiments_log", r"20251122_0925_Gen4_Code_Specialist", "final_adapter_gen4.pth")

print(f"🔧 [LIP Integration] Starting Universal Protocol Demo...")


class LIPSender:
    def __init__(self, adapter_path):
        print("\n📱 Initializing Client (TinyLlama)...")
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=DEVICE, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.model.eval()

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.LayerNorm(512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 4096),
            torch.nn.LayerNorm(4096)
        ).to(DEVICE)

        try:
            state = torch.load(adapter_path, map_location=DEVICE)
            # Mapping HeteroAdapter (named layers) to Sequential (indexed layers) if necessary
            # Ideally, define the class structure identically.
            try:
                self.adapter.load_state_dict(state)
            except:
                # Quick fix for structure mismatch between class and sequential
                from collections import OrderedDict
                new_state = OrderedDict()
                new_state['0.weight'] = state['encoder.0.weight']
                new_state['0.bias'] = state['encoder.0.bias']
                new_state['1.weight'] = state['encoder.1.weight']
                new_state['1.bias'] = state['encoder.1.bias']
                new_state['3.weight'] = state['decoder.0.weight']
                new_state['3.bias'] = state['decoder.0.bias']
                new_state['4.weight'] = state['decoder.1.weight']
                new_state['4.bias'] = state['decoder.1.bias']
                self.adapter.load_state_dict(new_state)

            print("✅ Adapter Loaded on Client.")
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            print("⚠️ Using random weights (Demo flow check only).")


    def create_packet(self, prompt):
        print(f"   Thinking: '{prompt}'...")
        with torch.no_grad():
            inp = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            vec_src = self.model(**inp, output_hidden_states=True).hidden_states[-1][:, -1, :].to(torch.float32)
            vec_trans = self.adapter(vec_src)
            packet = LIPPacket(vec_trans, source_model="TinyLlama+AdapterV1", intent="code")
            return packet.to_json()


class LIPReceiver:
    def __init__(self):
        print("\n☁️  Initializing Server (Llama-3)...")
        self.model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=DEVICE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
        except:
             self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=DEVICE, torch_dtype=torch.float32
            )
        self.model.eval()

        with torch.no_grad():
            self.ref_energy = self.model.get_input_embeddings().weight.norm(p=2, dim=-1).mean().item()
            print(f"⚡ Server Reference Energy: {self.ref_energy:.4f}")

    def process_packet(self, json_packet):

        print(f"📩 Packet Received. Size: {len(json_packet)} bytes.")
        packet = LIPPacket.from_json(json_packet)

        print(f"Origin: {packet.header.source_model} | ID: {packet.header.packet_id[:8]}...")

        vector = packet.unpack_vector(device=DEVICE)
        calibrated_vec = packet.calibrate_energy(self.ref_energy, vector)
        self._inject_and_generate(calibrated_vec)

    def _inject_and_generate(self, vector):
        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        bos_tensor = torch.tensor([[bos_id]], device=DEVICE)

        # Context Priming
        priming_text = "Here is the Python code snippet:\n"
        priming_ids = self.tokenizer(priming_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        with torch.no_grad():
            bos_embed = self.model.get_input_embeddings()(bos_tensor).to(self.model.dtype)
            priming_embed = self.model.get_input_embeddings()(priming_ids).to(self.model.dtype)
            vec_final = vector.unsqueeze(1).to(self.model.dtype)

            # Sandwich: [BOS] + [PRIMING] + [VECTOR]
            input_embeds = torch.cat([bos_embed, priming_embed, vec_final], dim=1)
            att_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=DEVICE)

            print(f"🤖 Context Activated: '{priming_text.strip()}'")
            print("🤖 Generating response...")

            gen_ids = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=att_mask,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

            output = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(f"\n🟢 LIP RESPONSE: {output}\n")


if __name__ == "__main__":
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter not found at {ADAPTER_PATH}. Please run 'src/train_gen4.py' first!")
        sys.exit(1)

    sender = LIPSender(ADAPTER_PATH)
    receiver = LIPReceiver()

    prompt = "write an python hello world function"

    print("-" * 60)
    json_payload = sender.create_packet(prompt)
    print(f"\n🌐 [NETWORK] Transmitting JSON: {json_payload[:100]}... [Encrypted]")
    receiver.process_packet(json_payload)
    print("-" * 60)